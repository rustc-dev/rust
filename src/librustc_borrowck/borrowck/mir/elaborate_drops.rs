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
use super::gather_moves::{MoveData, MovePathIndex, MovePathContent, Location};
use super::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use super::dataflow::{DataflowResults};
use super::{drop_flag_effects_for_location, on_all_children_bits};
use super::{DropFlagState};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{Subst, Substs, VecPerParamSpace};
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::middle::const_val::ConstVal;
use rustc::middle::lang_items;
use rustc::util::nodemap::FnvHashMap;
use rustc_mir::pretty;
use syntax::codemap::Span;

use std::fmt;
use std::iter;
use std::u32;

pub struct ElaborateDrops;

struct MirPatch<'tcx> {
    patch_map: Vec<Option<TerminatorKind<'tcx>>>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
    new_statements: Vec<(Location, StatementKind<'tcx>)>,
    new_temps: Vec<TempDecl<'tcx>>,
    pub resume_block: BasicBlock,
    next_temp: u32,
}

impl<'tcx> MirPatch<'tcx> {
    fn new(mir: &Mir<'tcx>) -> Self {
        let mut result = MirPatch {
            patch_map: iter::repeat(None)
                .take(mir.basic_blocks.len()).collect(),
            new_blocks: vec![],
            new_temps: vec![],
            new_statements: vec![],
            next_temp: mir.temp_decls.len() as u32,
            resume_block: START_BLOCK
        };

        let mut resume_block = None;
        let mut resume_stmt_block = None;
        for block in mir.all_basic_blocks() {
            let data = mir.basic_block_data(block);
            if let TerminatorKind::Resume = data.terminator().kind {
                if data.statements.len() > 0 {
                    resume_stmt_block = Some(block);
                } else {
                    resume_block = Some(block);
                }
                break
            }
        }
        let resume_block = resume_block.unwrap_or_else(|| {
            result.new_block(BasicBlockData {
                statements: vec![],
                terminator: Some(Terminator {
                    span: mir.span,
                    scope: ScopeId::new(0),
                    kind: TerminatorKind::Resume
                }),
                is_cleanup: true
            })});
        result.resume_block = resume_block;
        if let Some(resume_stmt_block) = resume_stmt_block {
            result.patch_terminator(resume_stmt_block, TerminatorKind::Goto {
                target: resume_block
            });
        }
        result
    }

    fn is_patched(&self, bb: BasicBlock) -> bool {
        self.patch_map[bb.index()].is_some()
    }

    fn terminator_loc(&self, mir: &Mir<'tcx>, bb: BasicBlock) -> Location {
        let offset = match bb.index().checked_sub(mir.basic_blocks.len()) {
            Some(index) => self.new_blocks[index].statements.len(),
            None => mir.basic_block_data(bb).statements.len()
        };
        Location {
            block: bb,
            index: offset
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
        let block = BasicBlock::new(self.patch_map.len());
        debug!("MirPatch: new_block: {:?}: {:?}", block, data);
        self.new_blocks.push(data);
        self.patch_map.push(None);
        block
    }

    fn patch_terminator(&mut self, block: BasicBlock, new: TerminatorKind<'tcx>) {
        assert!(self.patch_map[block.index()].is_none());
        debug!("MirPatch: patch_terminator({:?}, {:?})", block, new);
        self.patch_map[block.index()] = Some(new);
    }

    fn add_statement(&mut self, loc: Location, stmt: StatementKind<'tcx>) {
        debug!("MirPatch: add_statement({:?}, {:?})", loc, stmt);
        self.new_statements.push((loc, stmt));
    }

    fn add_assign(&mut self, loc: Location, lv: Lvalue<'tcx>, rv: Rvalue<'tcx>) {
        self.add_statement(loc, StatementKind::Assign(lv, rv));
    }

    fn apply(self, mir: &mut Mir<'tcx>) {
        debug!("MirPatch: {:?} new temps, starting from index {}: {:?}",
               self.new_temps.len(), mir.temp_decls.len(), self.new_temps);
        debug!("MirPatch: {} new blocks, starting from index {}",
               self.new_blocks.len(), mir.basic_blocks.len());
        mir.basic_blocks.extend(self.new_blocks);
        mir.temp_decls.extend(self.new_temps);
        for (src, patch) in self.patch_map.into_iter().enumerate() {
            if let Some(patch) = patch {
                debug!("MirPatch: patching block {:?}", src);
                mir.basic_blocks[src].terminator_mut().kind = patch;
            }
        }

        let mut new_statements = self.new_statements;
        new_statements.sort_by(|u,v| u.0.cmp(&v.0));

        let mut delta = 0;
        let mut last_bb = START_BLOCK;
        for (mut loc, stmt) in new_statements {
            if loc.block != last_bb {
                delta = 0;
                last_bb = loc.block;
            }
            debug!("MirPatch: adding statement {:?} at loc {:?}+{}",
                   stmt, loc, delta);
            loc.index += delta;
            let (span, scope) = Self::context_for_index(
                mir.basic_block_data(loc.block), loc
            );
            mir.basic_block_data_mut(loc.block).statements.insert(
                loc.index, Statement {
                    span: span,
                    scope: scope,
                    kind: stmt
                });
            delta += 1;
        }
    }

    fn context_for_index(data: &BasicBlockData, loc: Location) -> (Span, ScopeId) {
        match data.statements.get(loc.index) {
            Some(stmt) => (stmt.span, stmt.scope),
            None => (data.terminator().span, data.terminator().scope)
        }
    }

    fn context_for_location(&self, mir: &Mir, loc: Location) -> (Span, ScopeId) {
        let data = match loc.block.index().checked_sub(mir.basic_blocks.len()) {
            Some(new) => &self.new_blocks[new],
            None => mir.basic_block_data(loc.block)
        };
        Self::context_for_index(data, loc)
    }
}

impl<'tcx> MirPass<'tcx> for ElaborateDrops {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>)
    {
        debug!("elaborate_drops({:?} @ {:?})", src, mir.span);
        match src {
            MirSource::Fn(..) => {},
            _ => return
        }
        let id = src.item_id();
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);
        let move_data = MoveData::gather_moves(mir, tcx);
        let (elaborate_patch, _drop_flags) = {
            let mir = &*mir;
            let ((_, _, move_data), flow_inits) =
                super::do_dataflow(tcx, mir, id, &[], (tcx, mir, move_data),
                                   MaybeInitializedLvals::default());
            let ((_, _, move_data), flow_uninits) =
                super::do_dataflow(tcx, mir, id, &[], (tcx, mir, move_data),
                                   MaybeUninitializedLvals::default());

            match (tcx, mir, move_data) {
                ref ctxt => ElaborateDropsCtxt {
                    ctxt: ctxt,
                    param_env: &param_env,
                    flow_inits: flow_inits,
                    flow_uninits: flow_uninits,
                    drop_flags: FnvHashMap(),
                    patch: MirPatch::new(mir),
                }.elaborate()
            }
        };
        pretty::dump_mir(tcx, "elaborate_drops", &0, src, mir, None);
        elaborate_patch.apply(mir);
        pretty::dump_mir(tcx, "elaborate_drops", &1, src, mir, None);
    }
}

impl Pass for ElaborateDrops {}

struct InitializationData {
    live: Vec<usize>,
    dead: Vec<usize>
}

impl InitializationData {
    fn apply_location<'a,'tcx>(&mut self,
                               tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               mir: &Mir<'tcx>,
                               move_data: &MoveData<'tcx>,
                               loc: Location)
    {
        drop_flag_effects_for_location(tcx, mir, move_data, loc, |path, df| {
            debug!("at location {:?}: setting {:?} to {:?}",
                   loc, path, df);
            match df {
                DropFlagState::Live => {
                    self.live.set_bit(path.idx());
                    self.dead.clear_bit(path.idx());
                }
                DropFlagState::Dead => {
                    self.dead.set_bit(path.idx());
                    self.live.clear_bit(path.idx());
                }
            }
        });
    }

    fn state(&self, path: MovePathIndex) -> (bool, bool) {
        (self.live.get_bit(path.idx()), self.dead.get_bit(path.idx()))
    }
}

impl fmt::Debug for InitializationData {
    fn fmt(&self, _f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// What to do with the drop flags when we elaborate?
enum ElaborateKind {
    /// Clear them for this and all children
    Normal,
    /// Don't touch them
    RestHead,
    /// Clear them only for this
    RestTail,
}

struct ElaborateDropsCtxt<'a, 'tcx: 'a> {
    ctxt: &'a (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>),
    param_env: &'a ty::ParameterEnvironment<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'a, 'tcx>>,
    flow_uninits:  DataflowResults<MaybeUninitializedLvals<'a, 'tcx>>,
    drop_flags: FnvHashMap<MovePathIndex, u32>,
    patch: MirPatch<'tcx>,
}

#[derive(Copy, Clone, Debug)]
struct DropCtxt<'a, 'tcx: 'a> {
    span: Span,
    scope: ScopeId,
    is_cleanup: bool,

    init_data: &'a InitializationData,
    kind: ElaborateKind,

    lvalue: &'a Lvalue<'tcx>,
    path: MovePathIndex,
    succ: BasicBlock,
    unwind: Option<BasicBlock>
}

impl<'b, 'tcx> ElaborateDropsCtxt<'b, 'tcx> {
    fn tcx(&self) -> TyCtxt<'b, 'tcx, 'tcx> { self.ctxt.0 }
    fn mir(&self) -> &'b Mir<'tcx> { self.ctxt.1 }
    fn move_data(&self) -> &'b MoveData<'tcx> { &self.ctxt.2 }

    fn initialization_data_at(&self, loc: Location) -> InitializationData {
        let mut data = InitializationData {
            live: self.flow_inits.sets().on_entry_set_for(loc.block.index())
                .to_owned(),
            dead: self.flow_uninits.sets().on_entry_set_for(loc.block.index())
                .to_owned(),
        };
        for stmt in 0..loc.index {
            data.apply_location(self.ctxt.0, self.ctxt.1, &self.ctxt.2,
                                Location { block: loc.block, index: stmt });
        }
        data
    }

    fn create_drop_flag(&mut self, index: MovePathIndex) {
        let tcx = self.tcx();
        let patch = &mut self.patch;
        self.drop_flags.entry(index).or_insert_with(|| {
            patch.new_temp(tcx.types.bool)
        });
    }

    fn drop_flag(&mut self, index: MovePathIndex) -> Option<Lvalue<'tcx>> {
        self.drop_flags.get(&index).map(|t| Lvalue::Temp(*t))
    }

    fn elaborate(mut self) -> (MirPatch<'tcx>, FnvHashMap<MovePathIndex, u32>) {
        self.collect_drop_flags();
        self.elaborate_drops();

        self.drop_flags_on_init();
        self.drop_flags_for_fn_rets();
        self.drop_flags_for_args();
        self.drop_flags_for_locs();

        (self.patch, self.drop_flags)
    }

    fn path_needs_drop(&self, path: MovePathIndex) -> bool
    {
        match self.move_data().move_paths[path].content {
            MovePathContent::Lvalue(ref lvalue) => {
                let ty = self.mir().lvalue_ty(self.tcx(), lvalue)
                    .to_ty(self.tcx());
                debug!("path_needs_drop({:?}, {:?} : {:?})", path, lvalue, ty);

                self.tcx().type_needs_drop_given_env(ty, &self.param_env)
            }
            _ => false
        }
    }

    fn lvalue_concrete_strict_parent<'a>(&self, lv: &'a Lvalue<'tcx>)
                                         -> Option<&'a Lvalue<'tcx>>
    {
        if let &Lvalue::Projection(ref data) = lv {
            self.lvalue_concrete_parent(&data.base)
        } else {
            None
        }
    }

    fn lvalue_concrete_parent<'a>(&self, lv: &'a Lvalue<'tcx>)
                                  -> Option<&'a Lvalue<'tcx>>
    {
        let ty = self.mir().lvalue_ty(self.tcx(), lv)
            .to_ty(self.tcx());
        match ty.sty {
            ty::TyArray(..) | ty::TySlice(..) | ty::TyRef(..) | ty::TyRawPtr(..) => {
                Some(lv)
            }
            _ => match lv {
                &Lvalue::Projection(ref data) => {
                    self.lvalue_concrete_parent(&data.base)
                }
                _ => None
            }
        }
    }

    fn collect_drop_flags(&mut self)
    {
        for bb in self.mir().all_basic_blocks() {
            let data = self.mir().basic_block_data(bb);
            let terminator = data.terminator();
            let value = match terminator.kind {
                TerminatorKind::Drop { ref value, .. } => value,
                _ => continue
            };

            let init_data = self.initialization_data_at(Location {
                block: bb,
                index: data.statements.len()
            });

            let path = self.move_data().rev_lookup.find(value);
            debug!("collect_drop_flags: {:?}, lv {:?} (index {:?})",
                   bb, value, path);

            on_all_children_bits(self.tcx(), self.mir(), self.move_data(), path, |child| {
                if self.path_needs_drop(child) {
                    let (maybe_live, maybe_dead) = init_data.state(child);
                    debug!("collect_drop_flags: collecting {:?} from {:?}@{:?} - {:?}",
                           child, value, path, (maybe_live, maybe_dead));
                    if maybe_live && maybe_dead {
                        self.create_drop_flag(child)
                    }
                }
            });
        }
    }

    fn elaborate_drops(&mut self)
    {
        for bb in self.mir().all_basic_blocks() {
            let data = self.mir().basic_block_data(bb);
            let loc = Location { block: bb, index: data.statements.len() };
            let terminator = data.terminator();

            let resume_block = self.patch.resume_block;
            match terminator.kind {
                TerminatorKind::Drop { ref value, target, unwind } => {
                    let init_data = self.initialization_data_at(loc);
                    let path = self.move_data().rev_lookup.find(value);
                    self.elaborate_drop(&DropCtxt {
                        span: terminator.span,
                        scope: terminator.scope,
                        is_cleanup: data.is_cleanup,
                        init_data: &init_data,
                        lvalue: value,
                        kind: ElaborateKind::Normal,
                        path: path,
                        succ: target,
                        unwind: if data.is_cleanup {
                            None
                        } else {
                            Some(Option::unwrap_or(unwind, resume_block))
                        }
                    }, bb);
                }
                TerminatorKind::DropAndReplace { ref location, ref value,
                                                 target, unwind } =>
                {
                    assert!(!data.is_cleanup);

                    let unwind = unwind.unwrap_or_else(|| {
                        self.jump_to_resume_block(terminator.scope,
                                                  terminator.span)
                    });
                    let unwind = if data.is_cleanup {
                        None
                    } else {
                        Some(unwind)
                    };

                    if let Some(parent) = self.lvalue_concrete_strict_parent(location) {
                        // drop and replace behind a pointer/array/whatever. The location
                        // must be initialized.
                        debug!("elaborate_drop_and_replace({:?}) - parent = {:?}",
                               terminator, parent);
                        self.patch.patch_terminator(bb, TerminatorKind::Drop {
                            value: location.clone(),
                            target: target,
                            unwind: unwind
                        });
                    } else {
                        debug!("elaborate_drop_and_replace({:?})", terminator);
                        let init_data = self.initialization_data_at(loc);
                        let path = self.move_data().rev_lookup.find(location);

                        self.elaborate_drop(&DropCtxt {
                            span: terminator.span,
                            scope: terminator.scope,
                            is_cleanup: data.is_cleanup,
                            init_data: &init_data,
                            lvalue: location,
                            kind: ElaborateKind::Normal,
                            path: path,
                            succ: target,
                            unwind: unwind
                        }, bb);
                        on_all_children_bits(
                            self.tcx(), self.mir(), self.move_data(),
                            path, |child| {
                                self.set_drop_flag(loc, child, DropFlagState::Live)
                            });
                    }

                    self.patch.add_assign(Location { block: target, index: 0 },
                                          location.clone(), Rvalue::Use(value.clone()));
                    if let Some(unwind) = unwind {
                        self.patch.add_assign(Location { block: unwind, index: 0 },
                                              location.clone(), Rvalue::Use(value.clone()));
                    }
                }
                _ => continue
            };
        }
    }

    fn elaborate_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock) {
        debug!("elaborate_drop({:?})", c);

        let mut some_live = false;
        let mut some_dead = false;
        let mut children_count = 0;
        match c.kind {
            ElaborateKind::Normal => {
                on_all_children_bits(
                    self.tcx(), self.mir(), self.move_data(),
                    c.path, |child| {
                        if self.path_needs_drop(child) {
                            let (live, dead) = c.init_data.state(child);
                            debug!("elaborate_drop: state({:?}) = {:?}",
                                   child, (live, dead));
                            some_live |= live;
                            some_dead |= dead;
                            children_count += 1;
                        }
                    });
            }
            _ => {
                if self.path_needs_drop(c.path) {
                    let (live, dead) = c.init_data.state(c.path);
                    debug!("elaborate_drop: state({:?}) = {:?}",
                           c.path, (live, dead));
                    some_live |= live;
                    some_dead |= dead;
                    children_count += 1;
                }
            }
        }

        debug!("elaborate_drop({:?}): live - {:?}", c,
               (some_live, some_dead));
        match (some_live, some_dead) {
            (false, false) | (false, true) => {
                // dead drop - patch it out
                self.patch.patch_terminator(bb, TerminatorKind::Goto {
                    target: c.succ
                });
            }
            (true, false) => {
                // static drop - just set the flag
                self.patch.patch_terminator(bb, TerminatorKind::Drop {
                    value: c.lvalue.clone(),
                    target: c.succ,
                    unwind: c.unwind
                });
                self.drop_flags_for_drop(c, bb);
            }
            (true, true) => {
                // dynamic drop
                if children_count == 1 || self.must_complete_drop(c) {
                    self.conditional_drop(c, bb);
                } else {
                    self.partial_drop(c, bb);
                }
            }
        }
    }

    fn move_paths_for_fields(&self,
                             base_lv: Lvalue<'tcx>,
                             variant_path: MovePathIndex,
                             variant: ty::VariantDef<'tcx>,
                             substs: &'tcx Substs<'tcx>)
                             -> Vec<(Lvalue<'tcx>, Option<MovePathIndex>)>
    {
        let move_paths = &self.move_data().move_paths;
        variant.fields.iter().enumerate().map(|(i, f)| {
            let subpath =
                super::move_path_children_matching(move_paths, variant_path, |p| {
                    match p {
                        &Projection {
                            elem: ProjectionElem::Field(idx, _), ..
                        } => idx.index() == i,
                        _ => false
                    }
                });

            let field_ty =
                self.tcx().normalize_associated_type_in_env(
                    &f.ty(self.tcx(), substs),
                    &self.param_env
                );
            (base_lv.clone().field(Field::new(i), field_ty), subpath)
        }).collect()
    }

    fn drop_ladder<'a>(&mut self,
                       c: &DropCtxt<'a, 'tcx>,
                       unwind_ladder: Option<Vec<BasicBlock>>,
                       succ: BasicBlock,
                       fields: &[(Lvalue<'tcx>, Option<MovePathIndex>)],
                       is_cleanup: bool)
                       -> Vec<BasicBlock>
    {
        let mut succ = succ;
        let mut unwind_succ = if is_cleanup {
            None
        } else {
            c.unwind
        };
        let mut seen_self = false;

        fields.iter().rev().enumerate().map(|(i, &(ref lv, path))| {
            let field_c = DropCtxt {
                span: c.span,
                scope: c.scope,
                is_cleanup: is_cleanup,
                init_data: c.init_data,
                kind: if path.is_some() {
                    ElaborateKind::Normal
                } else if seen_self {
                    ElaborateKind::RestHead
                } else {
                    seen_self = true;
                    ElaborateKind::RestTail
                },
                lvalue: lv,
                path: path.unwrap_or(c.path),
                succ: succ,
                unwind: unwind_succ,
            };

            debug!("drop_ladder: for field {} ({:?})", i, lv);

            let drop_block = self.drop_block(&field_c);
            self.elaborate_drop(&field_c, drop_block);

            succ = drop_block;
            unwind_succ = unwind_ladder.as_ref().map(|p| p[i]);

            drop_block
        }).collect()
    }

    fn partial_drop_for_tuple<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock,
                                  tys: &[Ty<'tcx>])
    {
        debug!("partial_drop_for_tuple({:?}, {:?}, {:?})", c, bb, tys);

        let fields: Vec<_> = tys.iter().enumerate().map(|(i, &ty)| {
            (c.lvalue.clone().field(Field::new(i), ty),
             super::move_path_children_matching(
                 &self.move_data().move_paths, c.path, |proj| match proj {
                     &Projection {
                         elem: ProjectionElem::Field(f, _), ..
                     } => f.index() == i,
                     _ => false
                 }
            ))
        }).collect();

        let unwind_ladder = if c.is_cleanup {
            None
        } else {
            Some(self.drop_ladder(c, None, c.unwind.unwrap(), &fields, true))
        };

        let target =
            self.drop_ladder(c, unwind_ladder, c.succ, &fields, c.is_cleanup)
                .last().cloned().unwrap_or(c.succ);

        self.patch.patch_terminator(bb, TerminatorKind::Goto {
            target: target
        });
    }

    fn partial_drop_for_box<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock,
                                ty: Ty<'tcx>)
    {
        debug!("partial_drop_for_box({:?}, {:?}, {:?})", c, bb, ty);

        let inner = super::move_path_children_matching(
            &self.move_data().move_paths, c.path, |proj| match proj {
                &Projection { elem: ProjectionElem::Deref, .. } => true,
                _ => false
            }).unwrap();

        let deref = c.lvalue.clone().deref();
        let inner_c = DropCtxt {
            lvalue: &deref,
            unwind: c.unwind.map(|u| {
                self.free_block(c, ty, u, true)
            }),
            succ: self.free_block(c, ty, c.succ, c.is_cleanup),
            path: inner,
            ..*c
        };

        let free_inner = self.drop_block(&inner_c);
        self.elaborate_drop(&inner_c, free_inner);

        self.patch.patch_terminator(bb, TerminatorKind::Goto {
            target: free_inner
        });
    }

    fn partial_drop_for_variant<'a>(&mut self,
                                    c: &DropCtxt<'a, 'tcx>,
                                    drop_block: &mut Option<BasicBlock>,
                                    i: usize,
                                    v: ty::VariantDef<'tcx>,
                                    adt: ty::AdtDef<'tcx>,
                                    substs: &'tcx Substs<'tcx>)
                                    -> BasicBlock
    {
        let move_paths = &self.move_data().move_paths;

        let (base_lv, variant_path) = match adt.variants.len() {
            1 => (c.lvalue.clone(), c.path),
            _ => {
                let subpath = super::move_path_children_matching(
                    move_paths, c.path, |proj| match proj {
                        &Projection {
                            elem: ProjectionElem::Downcast(_, idx), ..
                        } => idx == i,
                        _ => false
                    });

                match subpath {
                    None => {
                        // variant not found - drop the entire enum
                        if let None = *drop_block {
                            let inner_c = DropCtxt {
                                kind: ElaborateKind::RestTail,
                                ..*c
                            };
                            let bb = self.drop_block(&inner_c);
                            self.elaborate_drop(&inner_c, bb);
                            *drop_block = Some(bb);
                        }
                        return drop_block.unwrap();
                    }
                    Some(subpath) => {
                        (c.lvalue.clone().elem(ProjectionElem::Downcast(adt, i)),
                         subpath)
                    }
                }
            }
        };

        let fields = self.move_paths_for_fields(base_lv, variant_path, v, substs);

        let unwind_ladder = if c.is_cleanup {
            None
        } else {
            Some(self.drop_ladder(c, None, c.unwind.unwrap(), &fields, true))
        };

        self.drop_ladder(c, unwind_ladder, c.succ, &fields, c.is_cleanup)
            .last().cloned().unwrap_or(c.succ)
    }

    fn partial_drop_for_adt<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock,
                                adt: ty::AdtDef<'tcx>, substs: &'tcx Substs<'tcx>) {
        debug!("partial_drop_for_adt({:?}, {:?}, {:?}, {:?})", c, bb, adt, substs);

        let mut drop_block = None;

        let variant_drops : Vec<BasicBlock> = adt.variants.iter().enumerate().map(|(i, v)| {
            self.partial_drop_for_variant(c, &mut drop_block, i, v, adt, substs)
        }).collect();

        match variant_drops.len() {
            1 => self.patch.patch_terminator(bb, TerminatorKind::Goto {
                target: variant_drops[0]
            }),
            _ => {
                // If there are multiple variants, then if something
                // is present within the enum the discriminant, tracked
                // by the rest path, must be initialized.
                //
                // Additionally, we do not want to switch on the
                // discriminant after it is free-ed, because that
                // way lies only trouble.

                let switch = TerminatorKind::Switch {
                    discr: c.lvalue.clone(),
                    adt_def: adt,
                    targets: variant_drops
                };

                if let Some(flag) = self.drop_flag(c.path) {
                    let switch_block = self.patch.new_block(BasicBlockData {
                        statements: vec![],
                        terminator: Some(Terminator {
                            scope: c.scope, span: c.span, kind: switch
                        }),
                        is_cleanup: c.is_cleanup
                    });

                    self.patch.patch_terminator(bb, TerminatorKind::If {
                        cond: Operand::Consume(flag),
                        targets: (switch_block, c.succ)
                    });
                } else {
                    self.patch.patch_terminator(bb, switch);
                }
            }
        }
    }

    fn partial_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock) {
        let ty = self.mir().lvalue_ty(self.tcx(), c.lvalue).to_ty(self.tcx());
        match ty.sty {
            ty::TyStruct(def, substs) | ty::TyEnum(def, substs) => {
                self.partial_drop_for_adt(c, bb, def, substs)
            }
            ty::TyTuple(tys) | ty::TyClosure(_, ty::ClosureSubsts {
                upvar_tys: tys, ..
            }) => {
                self.partial_drop_for_tuple(c, bb, tys);
            }
            ty::TyBox(ty) => {
                self.partial_drop_for_box(c, bb, ty);
            }
            _ => bug!("partial drop from non-ADT `{:?}`", ty)
        };
    }

    fn conditional_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock) {
        debug!("conditional_drop({:?}, {:?})", c, bb);
        let drop_bb = self.drop_block(c);
        self.drop_flags_for_drop(c, drop_bb);

        let flag = self.drop_flag(c.path).unwrap();
        self.patch.patch_terminator(bb, TerminatorKind::If {
            cond: Operand::Consume(flag),
            targets: (drop_bb, c.succ)
        });
    }

    fn drop_block<'a>(&mut self, c: &DropCtxt<'a, 'tcx>) -> BasicBlock {
        self.patch.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                scope: c.scope, span: c.span, kind: TerminatorKind::Drop {
                    value: c.lvalue.clone(),
                    target: c.succ,
                    unwind: c.unwind
                }
            }),
            is_cleanup: c.is_cleanup
        })
    }

    fn jump_to_resume_block<'a>(&mut self, scope: ScopeId, span: Span) -> BasicBlock {
        let resume_block = self.patch.resume_block;
        self.patch.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                scope: scope, span: span, kind: TerminatorKind::Goto {
                    target: resume_block
                }
            }),
            is_cleanup: true
        })
    }

    fn free_block<'a>(
        &mut self,
        c: &DropCtxt<'a, 'tcx>,
        ty: Ty<'tcx>,
        target: BasicBlock,
        is_cleanup: bool
    ) -> BasicBlock {
        let mut statements = vec![];
        if let Some(&flag) = self.drop_flags.get(&c.path) {
            statements.push(Statement {
                span: c.span,
                scope: c.scope,
                kind: StatementKind::Assign(
                    Lvalue::Temp(flag),
                    self.constant_bool(c.span, false)
                )
            });
        }

        let tcx = self.tcx();
        let unit_temp = Lvalue::Temp(self.patch.new_temp(tcx.mk_nil()));
        let free_func = tcx.lang_items.require(lang_items::BoxFreeFnLangItem)
            .unwrap_or_else(|e| tcx.sess.fatal(&e));
        let substs = tcx.mk_substs(Substs::new(
            VecPerParamSpace::new(vec![], vec![], vec![ty]),
            VecPerParamSpace::new(vec![], vec![], vec![])
        ));
        let fty = tcx.lookup_item_type(free_func).ty.subst(tcx, substs);

        self.patch.new_block(BasicBlockData {
            statements: statements,
            terminator: Some(Terminator {
                scope: c.scope, span: c.span, kind: TerminatorKind::Call {
                    func: Operand::Constant(Constant {
                        span: c.span,
                        ty: fty,
                        literal: Literal::Item {
                            def_id: free_func,
                            substs: substs
                        }
                    }),
                    args: vec![Operand::Consume(c.lvalue.clone())],
                    destination: Some((unit_temp, target)),
                    cleanup: None
                }
            }),
            is_cleanup: is_cleanup
        })
    }

    fn must_complete_drop<'a>(&self, c: &DropCtxt<'a, 'tcx>) -> bool {
        // if we have a destuctor, we must *not* split the drop.

        // dataflow can create unneeded children in some cases
        // - be sure to ignore them.

        let ty = self.mir().lvalue_ty(self.tcx(), c.lvalue).to_ty(self.tcx());

        match ty.sty {
            ty::TyStruct(def, _) | ty::TyEnum(def, _) => {
                def.has_dtor()
            }
            _ => false
        }
    }

    fn constant_bool(&self, span: Span, val: bool) -> Rvalue<'tcx> {
        Rvalue::Use(Operand::Constant(Constant {
            span: span,
            ty: self.tcx().types.bool,
            literal: Literal::Value { value: ConstVal::Bool(val) }
        }))
    }

    fn set_drop_flag(&mut self, loc: Location, path: MovePathIndex, val: DropFlagState) {
        if let Some(&flag) = self.drop_flags.get(&path) {
            let span = self.patch.context_for_location(self.mir(), loc).0;
            let val = self.constant_bool(span, val.value());
            self.patch.add_assign(loc, Lvalue::Temp(flag), val);
        }
    }

    fn drop_flags_on_init(&mut self) {
        let loc = Location { block: START_BLOCK, index: 0 };
        let span = self.patch.context_for_location(self.mir(), loc).0;
        let false_ = self.constant_bool(span, false);
        for flag in self.drop_flags.values() {
            self.patch.add_assign(loc, Lvalue::Temp(*flag), false_.clone());
        }
    }

    fn drop_flags_for_fn_rets(&mut self) {
        for bb in self.mir().all_basic_blocks() {
            let data = self.mir().basic_block_data(bb);
            if let TerminatorKind::Call {
                destination: Some((ref lv, tgt)), cleanup: Some(_), ..
            } = data.terminator().kind {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: tgt, index: 0 };
                let path = self.move_data().rev_lookup.find(lv);
                on_all_children_bits(
                    self.tcx(), self.mir(), self.move_data(), path,
                    |child| self.set_drop_flag(loc, child, DropFlagState::Live)
                );
            }
        }
    }

    fn drop_flags_for_args(&mut self) {
        let loc = Location { block: START_BLOCK, index: 0 };
        super::drop_flag_effects_for_function_entry(
            self.tcx(), self.mir(), self.move_data(), |path, ds| {
                self.set_drop_flag(loc, path, ds);
            }
        )
    }

    fn drop_flags_for_locs(&mut self) {
        // We intentionally iterate only over the *old* basic blocks,
        // the new basic blocks handle themselves
        for bb in self.mir().all_basic_blocks() {
            let data = self.mir().basic_block_data(bb);
            debug!("drop_flags_for_locs({:?})", data);
            for i in 0..(data.statements.len()+1) {
                debug!("drop_flag_for_locs: stmt {}", i);
                if i == data.statements.len() {
                    match data.terminator().kind {
                        TerminatorKind::Drop { .. } => {
                            // drop elaboration should handle that by itself
                            continue
                        }
                        TerminatorKind::DropAndReplace { .. } => {
                            // this only contains the use for the source
                            assert!(self.patch.is_patched(bb));                                         }
                        _ => {
                            assert!(!self.patch.is_patched(bb));
                        }
                    }
                }
                let loc = Location { block: bb, index: i };
                super::drop_flag_effects_for_location(
                    self.tcx(), self.mir(), self.move_data(), loc, |path, ds| {
                        self.set_drop_flag(loc, path, ds)
                    }
                )
            }

            // There may be a critical edge after this call,
            // so mark the return as initialized *before* the
            // call.
            if let TerminatorKind::Call {
                destination: Some((ref lv, _)), cleanup: None, ..
            } = data.terminator().kind {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: bb, index: data.statements.len() };
                let path = self.move_data().rev_lookup.find(lv);
                on_all_children_bits(
                    self.tcx(), self.mir(), self.move_data(), path,
                    |child| self.set_drop_flag(loc, child, DropFlagState::Live)
                );
            }
        }
    }

    fn drop_flags_for_drop<'a>(&mut self,
                               c: &DropCtxt<'a, 'tcx>,
                               bb: BasicBlock)
    {
        let loc = self.patch.terminator_loc(self.mir(), bb);
        match c.kind {
            ElaborateKind::Normal => {
                on_all_children_bits(
                    self.tcx(), self.mir(), self.move_data(), c.path,
                    |child| self.set_drop_flag(loc, child, DropFlagState::Dead)
                );
            }
            ElaborateKind::RestHead => {}
            ElaborateKind::RestTail => {
                self.set_drop_flag(loc, c.path, DropFlagState::Dead)
            }
        }
    }
}
