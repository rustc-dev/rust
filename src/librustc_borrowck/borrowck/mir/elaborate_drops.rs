// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use bitslice::BitSlice;
use super::gather_moves::{MoveData, MovePathIndex};
use super::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use super::dataflow::{DataflowResults};

use rustc::ty::{self, Ty, TyCtxt};
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::util::nodemap::FnvHashMap;
use rustc_mir::pretty;
use syntax::ast::NodeId;

use std::iter;
use std::u32;

pub struct ElaborateDrops;

struct MirPatch<'tcx> {
    patch_map: Vec<BasicBlock>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
    new_temps: Vec<TempDecl<'tcx>>,
    next_temp: u32,
}

impl<'tcx> MirPatch<'tcx> {
    fn new(mir: &Mir<'tcx>) -> Self {
        MirPatch {
            patch_map: iter::repeat(START_BLOCK)
                .take(mir.basic_blocks.len()).collect(),
            new_blocks: vec![],
            new_temps: vec![],
            next_temp: mir.temp_decls.len() as u32,
        }
    }

    fn new_temp(&mut self, ty: Ty<'tcx>) -> u32 {
        let index = self.next_temp;
        assert!(self.next_temp < u32::MAX);
        self.next_temp += 1;
        self.new_temps.push(TempDecl { ty: ty });
        index
    }

    fn new_block(&mut self, data: BasicBlockData<'tcx>) -> BasicBlock {
        let block = BasicBlock::new(self.patch_map.len() + self.new_blocks.len());
        self.new_blocks.push(data);
        block
    }

    fn patch_terminator(&mut self, block: BasicBlock, new: BasicBlock) {
        assert_eq!(self.patch_map[block.index()], START_BLOCK);
        self.patch_map[block.index()] = new;
    }

    fn apply(self, mir: &mut Mir<'tcx>) {
        debug!("MirPatch: {:?} new temps, starting from index {}: {:?}",
               self.new_temps.len(), mir.temp_decls.len(), self.new_temps);
        mir.basic_blocks.extend(self.new_blocks);
        mir.temp_decls.extend(self.new_temps);
        for (src, block) in self.patch_map.into_iter().enumerate() {
            if block == START_BLOCK { continue; }
            debug!("MirPatch: patching block {:?}", src);
            if let Some(ref mut terminator) = mir.basic_blocks[src].terminator {
                terminator.kind = TerminatorKind::Goto { target: block };
            }
        }
    }
}

impl<'tcx> MirPass<'tcx> for ElaborateDrops {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>)
    {
        debug!("elaborate_drops({:?})", src);
        match src {
            MirSource::Fn(..) if false => {},
            _ => return
        }
        let id = src.item_id();
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);
        let move_data = MoveData::gather_moves(mir, tcx);
        let (move_data, elaborate_patch) = {
            let mir = &*mir;
            let ((_, _, move_data), flow_inits) =
                super::do_dataflow(tcx, mir, id, &[], (tcx, mir, move_data),
                                   MaybeInitializedLvals::default());
            let ((_, _, move_data), flow_uninits) =
                super::do_dataflow(tcx, mir, id, &[], (tcx, mir, move_data),
                                   MaybeUninitializedLvals::default());

            ElaborateDropsCtxt {
                ctxt: (tcx, mir, move_data),
                param_env: param_env,
                flow_inits: flow_inits,
                flow_uninits: flow_uninits,
                drop_flags: FnvHashMap(),
                patch: MirPatch::new(mir),
            }.elaborate_drops()
        };
        pretty::dump_mir(tcx, "elaborate_drops", &0, src, mir, None);
        elaborate_patch.apply(mir);
        pretty::dump_mir(tcx, "elaborate_drops", &1, src, mir, None);
    }
}

impl Pass for ElaborateDrops {}

struct ElaborateDropsCtxt<'a, 'tcx: 'a> {
    ctxt: (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>),
    param_env: ty::ParameterEnvironment<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'a, 'tcx>>,
    flow_uninits:  DataflowResults<MaybeUninitializedLvals<'a, 'tcx>>,
    drop_flags: FnvHashMap<MovePathIndex, u32>,
    patch: MirPatch<'tcx>,
}

impl<'a, 'tcx> ElaborateDropsCtxt<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> { self.ctxt.0 }
    fn mir(&self) -> &'a Mir<'tcx> { self.ctxt.1 }
    fn move_data(&self) -> &MoveData<'tcx> { &self.ctxt.2 }

    fn drop_flag_for_index(&mut self, index: MovePathIndex) -> Lvalue<'tcx> {
        let tcx = self.tcx();
        let mut patch = &mut self.patch;
        Lvalue::Temp(*self.drop_flags.entry(index).or_insert_with(|| {
            patch.new_temp(tcx.types.bool)
        }))
    }
    fn elaborate_drops(mut self) -> (MoveData<'tcx>, MirPatch<'tcx>)
    {
        for bb in self.mir().all_basic_blocks() {
            let data = self.mir().basic_block_data(bb);
            let terminator = data.terminator();
            let (value, target, unwind) = match terminator.kind {
                TerminatorKind::Drop { ref value, target, unwind } => {
                    (value, target, unwind)
                }
                _ => continue
            };

            let index = self.move_data().rev_lookup.find(value);


//            let flag = self.drop_flag_for_index(index);
//            println!("found index {:?}, lv {:?}, flag {:?}", index, value, flag);

            let new_data = BasicBlockData {
                statements: vec![Statement {
                    span: terminator.span,
                    scope: terminator.scope,
                    kind: StatementKind::Assign(
                        Lvalue::Temp(0),
                        Rvalue::Use(Operand::Consume(Lvalue::Temp(0)))
                    )
                }],
                terminator: Some(Terminator {
                    kind: TerminatorKind::Drop {
                        value: value.clone(),
                        target: target,
                        unwind: unwind
                    }, ..*terminator
                }),
                is_cleanup: data.is_cleanup
            };
            let block = self.patch.new_block(new_data);
            self.patch.patch_terminator(bb, block);
        }

        (self.ctxt.2, self.patch)
    }
}
