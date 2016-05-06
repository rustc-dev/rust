use syntax::parse::token::keywords;
use syntax::codemap::{self, Span};

use rustc::mir::repr::{BasicBlock, BasicBlockData, Lvalue, Mir, TerminatorKind};
use rustc::session::Session;

use super::super::gather_moves::{MovePath, MovePathContent};
use super::AllSets;
use super::DataflowResults;
use super::BitDenotation;

pub fn issue_result_info<'tcx, O>(sess: &Session,
                                  mir: &Mir<'tcx>,
                                  flow_ctxt: &O::Ctxt,
                                  results: &DataflowResults<O>)
    where O: BitDenotation<Bit=MovePath<'tcx>>
{
    let info_warn = InfoWarn {
        sess: sess, mir: mir, flow_ctxt: flow_ctxt, results: results,
        render_elem: |elem, accum| {
            let arg_decl_debug_valid = |idx: usize| {
                mir.arg_decls[idx].debug_name != keywords::Invalid.name()
            };
            let s = |i:u32| i as usize;
            let rendered = match elem.content {
                MovePathContent::Lvalue(ref lval) => {
                    match *lval {
                        Lvalue::Var(idx) =>
                            format!("{:?}={}", lval, mir.var_decls[s(idx)].name),
                        Lvalue::Arg(idx) if arg_decl_debug_valid(s(idx)) =>
                            format!("{:?}={}", lval, mir.arg_decls[s(idx)].debug_name),
                        _ =>
                            format!("{:?}", lval),
                    }
                }
                MovePathContent::Static => "static".to_owned(),
            };
            accum.push_str(&rendered);
        }
    };
    info_warn.issue_result_info(results);
}

struct InfoWarn<'a, 'tcx: 'a, O: BitDenotation, R>
    where O: 'a, O::Ctxt: 'a, R: for <'b> Fn(&'b O::Bit, &'b mut String)
{
    sess: &'a Session,
    mir: &'a Mir<'tcx>,
    flow_ctxt: &'a O::Ctxt,
    results: &'a DataflowResults<O>,
    render_elem: R,
}

fn span_line<F>(codemap: &codemap::CodeMap, sp: Span, line_idx: F) -> Span
    where F: FnOnce(&codemap::FileLines) -> Option<usize>
{
    if let Ok(ref file_lines) = codemap.span_to_lines(sp) {
        if let Some(idx) = line_idx(file_lines) {
            let lines = file_lines.file.lines.borrow();
            return Span {
                lo: lines[idx],
                hi: lines[idx+1] - codemap::BytePos(1),
                expn_id: codemap::NO_EXPANSION,
            }
        }
    }
    return sp;
}

fn span_prior_line(codemap: &codemap::CodeMap, sp: Span) -> Span {
    span_line(codemap, sp, |file_lines| {
        file_lines.lines
            .first()
            .map(|line_info|line_info.line_index - 1)
    })
}

fn span_first_line(codemap: &codemap::CodeMap, sp: Span) -> Span {
    span_line(codemap, sp, |file_lines| {
        file_lines.lines
            .first()
            .map(|line_info|line_info.line_index)
    })
}

fn span_last_line(codemap: &codemap::CodeMap, sp: Span) -> Span {
    span_line(codemap, sp, |file_lines| {
        file_lines.lines
            .last()
            .map(|line_info|line_info.line_index)
    })
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Where { Entry, Exit }

impl<'a, 'tcx, O: BitDenotation, R> InfoWarn<'a, 'tcx, O, R>
    where R: for <'b> Fn(&'b O::Bit, &'b mut String)
{
    fn issue_result_info(&self, results: &DataflowResults<O>)
    {
        let mir = self.mir;
        let blocks = mir.all_basic_blocks();
        for bb in blocks {
            let bb_data = mir.basic_block_data(bb);
            self.issue_entry_info_for_block(results, bb, bb_data);
            self.issue_exit_info_for_block(results, bb, bb_data);
        }
    }

    fn issue_entry_info_for_block(&self,
                                  results: &DataflowResults<O>,
                                  bb: BasicBlock,
                                  bb_data: &BasicBlockData<'tcx>)
    {
        let all_sets = results.sets();
        let first_span = if bb_data.statements.len() > 0 {
            let span = bb_data.statements[0].span;
            span_prior_line(self.sess.codemap(), span)
        } else {
            let span = bb_data.terminator().span;
            // The spans for certain terminators can end up spanning
            // the whole function, which means using `span_prior_line`
            // points at a line that is nowhere near the conceptual
            // "end" of the function.
            //
            // As a hack to make output more intelligible, map such
            // spans to their last line rather than the preceding one.
            match bb_data.terminator().kind {
                TerminatorKind::Return { .. } => {
                    span_last_line(self.sess.codemap(), span)
                }
                _ => {
                    span_prior_line(self.sess.codemap(), span)
                }
            }
        };
        let entry_sets = all_sets.on_entry_set_for(bb.index());
        let interpreted = (results.0).interpret_set(self.flow_ctxt, &entry_sets[..]);
        let mut rendered = String::new();

        rendered.push_str("entry");
        self.push_rendered_elem_strs(&interpreted[..], &mut rendered);
        self.sess.span_err(first_span, &rendered);
    }

    fn issue_exit_info_for_block(&self,
                                 results: &DataflowResults<O>,
                                 bb: BasicBlock,
                                 bb_data: &BasicBlockData<'tcx>)
    {
        // Trying to extract a span representing the end of the basic
        // block is a little weird. For example, the span for the If
        // terminator of a basic block covers the *whole* `if`
        // expression, not just the condition expression. So if you
        // jump to the end of that span for the whole `if`, that is
        // probably a fairly misleading place to insert a dataflow
        // annotation for the If terminator itself.
        //
        // The right thing may be to dispatch on the type of
        // terminator. But for now, as a silly hack, lets use the
        // first line of the terminator.
        let all_sets = results.sets();
        let term_span = bb_data.terminator().span;
        let end_of_span = match bb_data.terminator().kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume { .. } |
            TerminatorKind::Return { .. } |
            TerminatorKind::Drop { .. } |
            TerminatorKind::Call { .. } => {
                span_last_line(self.sess.codemap(), term_span)
            }
            TerminatorKind::If { .. } |
            TerminatorKind::Switch { .. } |
            TerminatorKind::SwitchInt { .. } => {
                span_first_line(self.sess.codemap(), term_span)
            }
        };
        debug!("term: {:?} term_span {:?} end_of_span: {:?}",
               bb_data.terminator(), term_span, end_of_span);

        let exit_sets = all_sets.on_exit_set_for(bb.index());
        let interpreted = (results.0).interpret_set(self.flow_ctxt, &exit_sets[..]);
        let mut rendered = String::new();

        rendered.push_str("exit:");
        rendered.push_str(match bb_data.terminator().kind {
            TerminatorKind::Goto { .. }      => "goto      ",
            TerminatorKind::Resume { .. }    => "resume    ",
            TerminatorKind::Return { .. }    => "return    ",
            TerminatorKind::Drop { .. }      => "drop      ",
            TerminatorKind::Call { .. }      => "call      ",
            TerminatorKind::If { .. }        => "if        ",
            TerminatorKind::Switch { .. }    => "switch    ",
            TerminatorKind::SwitchInt { .. } => "switch_int",
        });

        self.push_rendered_elem_strs(&interpreted[..], &mut rendered);
        self.sess.span_err(end_of_span, &rendered);
    }

    fn push_rendered_elem_strs(&self, interpreted: &[&O::Bit], rendered: &mut String) {
        rendered.push_str(" dataflow:");
        rendered.push_str(O::name());
        rendered.push_str(" [");
        let mut saw_one = false;
        for elem in interpreted {
            if saw_one { rendered.push_str("|"); }
            (self.render_elem)(elem, rendered);
            saw_one = true;
        }
        rendered.push_str("]");
    }
}
