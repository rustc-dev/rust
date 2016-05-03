use syntax::parse::token::keywords;

use rustc::mir::repr::{Lvalue, Mir};
use rustc::session::Session;

use super::super::gather_moves::{MovePath, MovePathContent};
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
    };
    info_warn.issue_result_info(results, |elem, accum| {
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
    });
}

struct InfoWarn<'a, 'tcx: 'a, O: BitDenotation>
    where O: 'a, O::Ctxt: 'a
{
    sess: &'a Session,
    mir: &'a Mir<'tcx>,
    flow_ctxt: &'a O::Ctxt,
    results: &'a DataflowResults<O>,
}

impl<'a, 'tcx, O: BitDenotation> InfoWarn<'a, 'tcx, O> {
    fn issue_result_info<R>(&self,
                            results: &DataflowResults<O>,
                            render_elem: R)
        where R: Fn(&O::Bit, &mut String)
    {
        let mir = self.mir;
        let blocks = mir.all_basic_blocks();
        let all_sets = results.sets();
        for bb in blocks {
            let bb_data = mir.basic_block_data(bb);
            let first_span = if bb_data.statements.len() > 0 {
                bb_data.statements[0].span
            } else {
                bb_data.terminator
                    .as_ref()
                    .expect("nil statmements implies terminator present")
                    .span
            };

            let sets = all_sets.on_entry_set_for(bb.index());
            let interpreted = (results.0).interpret_set(self.flow_ctxt, sets);
            let mut rendered = String::new();
            let mut saw_one = false;
            rendered.push_str(O::name());
            rendered.push_str(" [");
            for elem in &interpreted {
                if saw_one { rendered.push_str("|"); }
                render_elem(elem, &mut rendered);
                saw_one = true;
            }
            rendered.push_str("]");
            self.sess.span_err(first_span, &rendered);
        }
    }
}
