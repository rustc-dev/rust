// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::gather_moves::MoveData;
use super::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use super::dataflow::{DataflowResults};

use rustc::ty::{self, TyCtxt};
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc_mir::pretty;
use syntax::ast::NodeId;

use std::iter;

pub struct ElaborateDrops;

struct MirPatch<'tcx> {
    patch_map: Vec<BasicBlock>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
}

impl<'tcx> MirPatch<'tcx> {
    fn new(mir: &Mir<'tcx>) -> Self {
        MirPatch {
            patch_map: iter::repeat(START_BLOCK)
                .take(mir.basic_blocks.len()).collect(),
            new_blocks: Vec::new()
        }
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
        mir.basic_blocks.extend(self.new_blocks);
        for (src, block) in self.patch_map.into_iter().enumerate() {
            if block == START_BLOCK { continue; }
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
            MirSource::Fn(..) => {},
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
            let (move_data, flow_uninits) =
                super::do_dataflow(tcx, mir, id, &[], move_data,
                                   MaybeUninitializedLvals::default());

            ElaborateDropsCtxt {
                tcx: tcx,
                param_env: param_env,
                mir: mir,
                move_data: move_data,
                flow_inits: flow_inits,
                flow_uninits: flow_uninits,
            }.elaborate_drops()
        };
        elaborate_patch.apply(mir);
        pretty::dump_mir(tcx, "elaborate_drops", &0, src, mir, None);
    }
}

impl Pass for ElaborateDrops {}

struct ElaborateDropsCtxt<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParameterEnvironment<'tcx>,
    mir: &'a Mir<'tcx>,
    move_data: MoveData<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'a, 'tcx>>,
    flow_uninits:  DataflowResults<MaybeUninitializedLvals<'tcx>>,
}

impl<'a, 'tcx> ElaborateDropsCtxt<'a, 'tcx> {
    fn elaborate_drops(mut self) -> (MoveData<'tcx>, MirPatch<'tcx>)
    {
        let mut patch = MirPatch::new(self.mir);

        for bb in self.mir.all_basic_blocks() {
            let data = self.mir.basic_block_data(bb);
            let terminator = data.terminator.as_ref().unwrap();
            let (value, target, unwind) = match terminator.kind {
                TerminatorKind::Drop { ref value, target, unwind } => {
                    (value, target, unwind)
                }
                _ => continue
            };
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
        }

        (self.move_data, patch)
    }
}
