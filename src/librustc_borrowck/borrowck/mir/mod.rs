// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrowck::BorrowckCtxt;

use syntax::ast::{self, MetaItem};
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::ptr::P;

use rustc::hir;
use rustc::hir::intravisit::{FnKind};

use rustc::mir::repr::{BasicBlock, BasicBlockData, Mir, Statement, Terminator};
use rustc::session::Session;

mod abs_domain;
mod dataflow;
mod gather_moves;
// mod graphviz;

use self::dataflow::{BitDenotation};
use self::dataflow::{Dataflow, DataflowAnalysis, DataflowResults};
use self::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use self::gather_moves::{MoveData};

use std::fmt::Debug;

#[derive(Debug)]
pub struct BorrowckMirData<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub flow_inits: DataflowResults<MaybeInitializedLvals<'tcx>>,
    pub flow_uninits: DataflowResults<MaybeUninitializedLvals<'tcx>>,
}

fn has_rustc_mir_with(attrs: &[ast::Attribute], name: &str) -> Option<P<MetaItem>> {
    for attr in attrs {
        if attr.check_name("rustc_mir") {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                if item.check_name(name) {
                    return Some(item.clone())
                }
            }
        }
    }
    return None;
}

pub fn borrowck_mir<'b, 'a: 'b, 'tcx: 'a>(
    bcx: &'b mut BorrowckCtxt<'a, 'tcx>,
    fk: FnKind,
    _decl: &hir::FnDecl,
    mir: &'a Mir<'tcx>,
    body: &hir::Block,
    _sp: Span,
    id: ast::NodeId,
    attributes: &[ast::Attribute]) {
    match fk {
        FnKind::ItemFn(name, _, _, _, _, _, _) |
        FnKind::Method(name, _, _, _) => {
            debug!("borrowck_mir({}) UNIMPLEMENTED", name);
        }
        FnKind::Closure(_) => {
            debug!("borrowck_mir closure (body.id={}) UNIMPLEMENTED", body.id);
        }
    }

    let move_data = MoveData::gather_moves(mir, bcx.tcx);
    let (move_data, flow_inits) =
        do_dataflow(bcx, mir, id, attributes, move_data, MaybeInitializedLvals::default());
    let (move_data, flow_uninits) =
        do_dataflow(bcx, mir, id, attributes, move_data, MaybeUninitializedLvals::default());

    if has_rustc_mir_with(attributes, "dataflow_info_maybe_init").is_some() {
        dataflow::issue_result_info(bcx.tcx.sess,
                                    mir,
                                    &move_data,
                                    &flow_inits);
    }
    if has_rustc_mir_with(attributes, "dataflow_info_maybe_uninit").is_some() {
        dataflow::issue_result_info(bcx.tcx.sess,
                                    mir,
                                    &move_data,
                                    &flow_uninits);
    }

    let mut mbcx = MirBorrowckCtxt {
        bcx: bcx,
        mir: mir,
        node_id: id,
        move_data: move_data,
        flow_inits: flow_inits,
        flow_uninits: flow_uninits,
    };

    for bb in mir.all_basic_blocks() {
        mbcx.process_basic_block(bb);
    }

    debug!("borrowck_mir done");

    fn do_dataflow<'a, 'tcx, BD>(bcx: &mut BorrowckCtxt<'a, 'tcx>,
                                 mir: &'a Mir<'tcx>,
                                 node_id: ast::NodeId,
                                 attributes: &[ast::Attribute],
                                 move_data: MoveData<'tcx>,
                                 bd: BD) -> (MoveData<'tcx>, DataflowResults<BD>)
        where BD: BitDenotation<Ctxt=MoveData<'tcx>>, BD::Bit: Debug
    {
        use syntax::attr::AttrMetaMethods;

        let name_found = |sess: &Session, attrs: &[ast::Attribute], name| -> Option<String> {
            if let Some(item) = has_rustc_mir_with(attrs, name) {
                if let Some(s) = item.value_str() {
                    return Some(s.to_string())
                } else {
                    sess.span_err(
                        item.span,
                        &format!("{} attribute requires a path", item.name()));
                    return None;
                }
            }
            return None;
        };

        let print_preflow_to =
            name_found(bcx.tcx.sess, attributes, "borrowck_graphviz_preflow");
        let print_postflow_to =
            name_found(bcx.tcx.sess, attributes, "borrowck_graphviz_postflow");

        let mut mbcx = MirBorrowckCtxtPreDataflow {
            bcx: bcx,
            mir: mir,
            node_id: node_id,
            print_preflow_to: print_preflow_to,
            print_postflow_to: print_postflow_to,
            flow_state: DataflowAnalysis::new(mir, move_data, bd),
        };

        mbcx.dataflow();
        mbcx.flow_state.results()
    }
}

pub struct MirBorrowckCtxtPreDataflow<'b, 'a: 'b, 'tcx: 'a, BD>
    where BD: BitDenotation<Ctxt=MoveData<'tcx>>
{
    bcx: &'b mut BorrowckCtxt<'a, 'tcx>,
    mir: &'b Mir<'tcx>,
    node_id: ast::NodeId,
    flow_state: DataflowAnalysis<'a, 'tcx, BD>,
    print_preflow_to: Option<String>,
    print_postflow_to: Option<String>,
}

pub struct MirBorrowckCtxt<'b, 'a: 'b, 'tcx: 'a> {
    bcx: &'b mut BorrowckCtxt<'a, 'tcx>,
    mir: &'b Mir<'tcx>,
    node_id: ast::NodeId,
    move_data: MoveData<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'tcx>>,
    flow_uninits: DataflowResults<MaybeUninitializedLvals<'tcx>>
}

impl<'b, 'a: 'b, 'tcx: 'a> MirBorrowckCtxt<'b, 'a, 'tcx> {
    fn process_basic_block(&mut self, bb: BasicBlock) {
        let &BasicBlockData { ref statements, ref terminator, is_cleanup: _ } =
            self.mir.basic_block_data(bb);
        for stmt in statements {
            self.process_statement(bb, stmt);
        }

        self.process_terminator(bb, terminator);
    }

    fn process_statement(&mut self, bb: BasicBlock, stmt: &Statement<'tcx>) {
        debug!("MirBorrowckCtxt::process_statement({:?}, {:?}", bb, stmt);
    }

    fn process_terminator(&mut self, bb: BasicBlock, term: &Option<Terminator<'tcx>>) {
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?})", bb, term);
    }
}
