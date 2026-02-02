# Berkeley FP -- 21st Century Edition
# Python + Streamlit reimplementation of the Berkeley FP interpreter (4.2BSD)
#
# Operators:
#   f : x            apply f to x
#   f @ g            composition
#   &f               apply-to-all
#   !f / !f(z)       right-insert (foldr)
#   \!f / \!f(z)     left-insert (foldl)
#   [f g h]          construction
#   p -> f ; g       condition
#   %x               constant function (~ also accepted)
#   n                selector (1, 2, ...)

import streamlit as st
from dataclasses import dataclass
from typing import Any

APP_VERSION = 13

# ============================================================
# DATA MODEL
# ============================================================

BOTTOM = object()

class FPError(Exception):
    pass

def is_atom(v):
    return isinstance(v, (int, float, str, bool)) or v is BOTTOM

def is_sequence(v):
    return isinstance(v, list)

def fp_equal(a, b):
    if is_atom(a) and is_atom(b):
        return a == b
    if is_sequence(a) and is_sequence(b):
        if len(a) != len(b):
            return False
        return all(fp_equal(x, y) for x, y in zip(a, b))
    return False

def fp_format(v):
    if v is BOTTOM:
        return "?"
    if v is True:
        return "T"
    if v is False:
        return "F"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else str(v)
    if isinstance(v, (int, str)):
        return str(v)
    if is_sequence(v):
        return "<{}>".format(" ".join(fp_format(x) for x in v))
    return str(v)

# ============================================================
# TOKENIZER
# ============================================================

TK_NUM = 'NUM'; TK_SYM = 'SYM'
TK_LANGLE = '<'; TK_RANGLE = '>'
TK_LBRACK = '['; TK_RBRACK = ']'
TK_LPAREN = '('; TK_RPAREN = ')'
TK_COLON = ':';  TK_AT = '@'
TK_AMP = '&';    TK_BANG = '!'
TK_BACKBANG = 'BACKBANG'
TK_ARROW = '->'; TK_SEMI = ';'
TK_COMMA = ',';  TK_TILDE = '~';  TK_PERCENT = '%'
TK_LBRACE = '{'; TK_RBRACE = '}'
TK_EOF = 'EOF'

@dataclass
class Token:
    type: str
    value: Any
    pos: int

def tokenize(src):
    tokens = []
    i = 0
    n = len(src)
    while i < n:
        if src[i].isspace():
            i += 1; continue
        if i + 1 < n and src[i:i+2] == '--':
            break
        if i + 1 < n and src[i] == '\\' and src[i+1] == '!':
            tokens.append(Token(TK_BACKBANG, '\\', i)); i += 2; continue
        if src[i] == '\\':
            # bare backslash = left-insert (same as \!)
            tokens.append(Token(TK_BACKBANG, '\\', i)); i += 1; continue
        if i + 1 < n and src[i:i+2] == '->':
            tokens.append(Token(TK_ARROW, '->', i)); i += 2; continue
        simple = {
            '{': TK_LBRACE, '}': TK_RBRACE, '<': TK_LANGLE, '>': TK_RANGLE,
            '[': TK_LBRACK, ']': TK_RBRACK, '(': TK_LPAREN, ')': TK_RPAREN,
            ':': TK_COLON, '@': TK_AT, '&': TK_AMP, '!': TK_BANG,
            ';': TK_SEMI, ',': TK_COMMA, '~': TK_TILDE, '%': TK_PERCENT,
        }
        if src[i] in simple:
            tokens.append(Token(simple[src[i]], src[i], i)); i += 1; continue
        if src[i].isdigit() or (
            src[i] == '-' and i + 1 < n and src[i+1].isdigit()
            and (not tokens or tokens[-1].type != TK_RPAREN)
        ):
            j = i
            if src[i] == '-': i += 1
            while i < n and src[i].isdigit(): i += 1
            if i < n and src[i] == '.':
                i += 1
                while i < n and src[i].isdigit(): i += 1
                tokens.append(Token(TK_NUM, float(src[j:i]), j))
            else:
                tokens.append(Token(TK_NUM, int(src[j:i]), j))
            continue
        if src[i].isalpha() or src[i] == '_':
            j = i
            while i < n and (src[i].isalnum() or src[i] == '_'): i += 1
            tokens.append(Token(TK_SYM, src[j:i], j)); continue
        if src[i] in '+-*/=':
            tokens.append(Token(TK_SYM, src[i], i)); i += 1; continue
        raise FPError("syntax error: unexpected '{}' at position {}".format(src[i], i))
    tokens.append(Token(TK_EOF, None, i))
    return tokens

# ============================================================
# AST NODES
# ============================================================

@dataclass
class ASTAtom:
    value: Any
@dataclass
class ASTSequence:
    elements: list
@dataclass
class ASTApply:
    func: Any; arg: Any
@dataclass
class ASTCompose:
    left: Any; right: Any
@dataclass
class ASTConstruct:
    funcs: list
@dataclass
class ASTApplyAll:
    func: Any
@dataclass
class ASTInsertR:
    func: Any; seed: Any
@dataclass
class ASTInsertL:
    func: Any; seed: Any
@dataclass
class ASTCondition:
    pred: Any; then_f: Any; else_f: Any
@dataclass
class ASTConstant:
    value: Any
@dataclass
class ASTFuncRef:
    name: str
@dataclass
class ASTDef:
    name: str; body: Any

# ============================================================
# PARSER
# ============================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens; self.pos = 0
    def peek(self):
        return self.tokens[self.pos]
    def advance(self):
        t = self.tokens[self.pos]; self.pos += 1; return t
    def expect(self, tt):
        t = self.peek()
        if t.type != tt:
            raise FPError("syntax error: expected '{}', got '{}' at pos {}".format(tt, t.value, t.pos))
        return self.advance()
    def at(self, *tts):
        return self.peek().type in tts

    def parse_top(self):
        if self.at(TK_LBRACE):
            return self.parse_definition()
        result = self.parse_expr()
        if not self.at(TK_EOF):
            raise FPError("syntax error: unexpected '{}' at pos {}".format(self.peek().value, self.peek().pos))
        return result

    def parse_definition(self):
        self.expect(TK_LBRACE)
        name_tok = self.expect(TK_SYM)
        body = self.parse_func_expr()
        self.expect(TK_RBRACE)
        return ASTDef(name_tok.value, body)

    def parse_expr(self):
        left = self.parse_func_expr()
        if self.at(TK_COLON):
            self.advance()
            right = self.parse_expr()
            return ASTApply(left, right)
        return left

    def parse_func_expr(self):
        return self.parse_cond_expr()

    def parse_cond_expr(self):
        left = self.parse_comp_expr()
        if self.at(TK_ARROW):
            self.advance()
            then_f = self.parse_comp_expr()
            self.expect(TK_SEMI)
            else_f = self.parse_cond_expr()
            return ASTCondition(left, then_f, else_f)
        return left

    def parse_comp_expr(self):
        left = self.parse_prefix_expr()
        while self.at(TK_AT):
            self.advance()
            right = self.parse_prefix_expr()
            left = ASTCompose(left, right)
        return left

    def parse_prefix_expr(self):
        if self.at(TK_AMP):
            self.advance()
            return ASTApplyAll(self._amp_operand())
        if self.at(TK_BANG):
            self.advance()
            op = self._tight_prefix(); seed = self._try_seed()
            return ASTInsertR(op, seed)
        if self.at(TK_BACKBANG):
            self.advance()
            op = self._tight_prefix(); seed = self._try_seed()
            return ASTInsertL(op, seed)
        if self.at(TK_PERCENT) or self.at(TK_TILDE):
            self.advance()
            return ASTConstant(self.parse_atom_expr())
        return self.parse_atom_expr()

    def _tight_prefix(self):
        if self.at(TK_AMP):
            self.advance(); return ASTApplyAll(self._amp_operand())
        if self.at(TK_BANG):
            self.advance(); op = self._tight_prefix(); return ASTInsertR(op, self._try_seed())
        if self.at(TK_BACKBANG):
            self.advance(); op = self._tight_prefix(); return ASTInsertL(op, self._try_seed())
        if self.at(TK_PERCENT) or self.at(TK_TILDE):
            self.advance(); return ASTConstant(self.parse_atom_expr())
        return self.parse_atom_expr()

    def _amp_operand(self):
        left = self._tight_prefix()
        while self.at(TK_AT):
            self.advance(); left = ASTCompose(left, self._tight_prefix())
        return left

    def _try_seed(self):
        if self.at(TK_LPAREN):
            self.advance(); s = self.parse_expr(); self.expect(TK_RPAREN); return s
        return None

    def parse_atom_expr(self):
        t = self.peek()
        if t.type == TK_NUM:
            self.advance(); return ASTAtom(t.value)
        if t.type == TK_SYM:
            self.advance()
            if t.value == 'T': return ASTAtom(True)
            if t.value == 'F': return ASTAtom(False)
            return ASTFuncRef(t.value)
        if t.type == TK_LPAREN:
            self.advance(); inner = self.parse_expr(); self.expect(TK_RPAREN); return inner
        if t.type == TK_LBRACK:
            self.advance()
            funcs = []
            while not self.at(TK_RBRACK, TK_EOF):
                funcs.append(self.parse_func_expr())
                if self.at(TK_COMMA): self.advance()
            self.expect(TK_RBRACK); return ASTConstruct(funcs)
        if t.type == TK_LANGLE:
            self.advance()
            elems = []
            while not self.at(TK_RANGLE, TK_EOF):
                elems.append(self.parse_expr())
                if self.at(TK_COMMA): self.advance()
            self.expect(TK_RANGLE); return ASTSequence(elems)
        raise FPError("syntax error: unexpected '{}' at pos {}".format(t.value, t.pos))

def parse(src):
    return Parser(tokenize(src)).parse_top()

# ============================================================
# EVALUATOR  (class-name dispatch for Streamlit hot-reload safety)
# ============================================================

MAX_STEPS = 100000

class Evaluator:
    def __init__(self):
        self.user_funcs = {}
        self.steps = 0
        self._output_buffer = []

    def reset_steps(self):
        self.steps = 0

    def _ck(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise FPError("non-terminating")

    def eval_expr(self, node):
        self._ck()
        cn = type(node).__name__
        if cn == 'ASTAtom':       return node.value
        if cn == 'ASTSequence':   return [self.eval_expr(e) for e in node.elements]
        if cn == 'ASTApply':      return self._apply_val(node.func, self.eval_expr(node.arg))
        # A bare function ref without application: treat as symbol atom
        # (In FP, symbols are values when not in function position)
        if cn == 'ASTFuncRef':    return node.name
        # These remain as function objects (only meaningful in function position)
        if cn in ('ASTCompose','ASTConstruct','ASTApplyAll',
                  'ASTInsertR','ASTInsertL','ASTCondition','ASTConstant'):
            return node
        raise FPError("eval error: cannot evaluate {}".format(cn))

    def _apply_val(self, fn, val):
        self._ck()
        cn = type(fn).__name__

        if cn == 'ASTFuncRef':
            nm = fn.name
            if nm in PRIMITIVES:    return PRIMITIVES[nm](val, self)
            if nm in self.user_funcs: return self._apply_val(self.user_funcs[nm], val)
            raise FPError("undefined: {}".format(nm))

        if cn == 'ASTAtom':
            v = fn.value
            if isinstance(v, int):
                if not is_sequence(val):
                    raise FPError("error: selector expects sequence")
                if v > 0:
                    if v > len(val): raise FPError("error: selector {} out of range".format(v))
                    return val[v - 1]
                if v < 0:
                    if abs(v) > len(val): raise FPError("error: selector {} out of range".format(v))
                    return val[v]
                raise FPError("error: selector 0 undefined")
            raise FPError("error: {} is not a function".format(fp_format(v)))

        if cn == 'ASTCompose':
            return self._apply_val(fn.left, self._apply_val(fn.right, val))

        if cn == 'ASTConstruct':
            return [self._apply_val(f, val) for f in fn.funcs]

        if cn == 'ASTApplyAll':
            if not is_sequence(val): raise FPError("error: & expects sequence")
            return [self._apply_val(fn.func, v) for v in val]

        if cn == 'ASTInsertR':
            if not is_sequence(val): raise FPError("error: ! expects sequence")
            f = fn.func
            if fn.seed is not None:
                r = self.eval_expr(fn.seed)
                for x in reversed(val): r = self._apply_val(f, [x, r])
                return r
            if len(val) == 0: raise FPError("error: ! on empty sequence without seed")
            if len(val) == 1: return val[0]
            r = val[-1]
            for x in reversed(val[:-1]): r = self._apply_val(f, [x, r])
            return r

        if cn == 'ASTInsertL':
            if not is_sequence(val): raise FPError("error: \\ expects sequence")
            f = fn.func
            if fn.seed is not None:
                r = self.eval_expr(fn.seed)
                for x in val: r = self._apply_val(f, [r, x])
                return r
            if len(val) == 0: raise FPError("error: \\ on empty sequence without seed")
            if len(val) == 1: return val[0]
            r = val[0]
            for x in val[1:]: r = self._apply_val(f, [r, x])
            return r

        if cn == 'ASTCondition':
            pv = self._apply_val(fn.pred, val)
            if pv is True:  return self._apply_val(fn.then_f, val)
            if pv is False: return self._apply_val(fn.else_f, val)
            raise FPError("error: condition predicate must return T or F")

        if cn == 'ASTConstant':
            return self.eval_expr(fn.value)

        raise FPError("error: cannot apply {}".format(cn))

# ============================================================
# PRIMITIVE FUNCTIONS
# ============================================================

def _rseq(v, n):
    if not is_sequence(v): raise FPError("error: {} expects sequence".format(n))
    return v
def _rpair(v, n):
    s = _rseq(v, n)
    if len(s) != 2: raise FPError("error: {} expects pair".format(n))
    return s[0], s[1]
def _rneseq(v, n):
    s = _rseq(v, n)
    if not s: raise FPError("error: {} expects non-empty sequence".format(n))
    return s

def _arith(n, op):
    def f(v, ev):
        a, b = _rpair(v, n)
        if not isinstance(a, (int,float)) or not isinstance(b, (int,float)):
            raise FPError("error: {} expects numeric pair".format(n))
        try: return op(a, b)
        except ZeroDivisionError: raise FPError("error: division by zero")
    return f

def _cmp(n, op):
    def f(v, ev):
        a, b = _rpair(v, n)
        if not isinstance(a, (int,float)) or not isinstance(b, (int,float)):
            raise FPError("error: {} expects numeric pair".format(n))
        return op(a, b)
    return f

PRIMITIVES = {
    'first': lambda v,e: _rneseq(v,"first")[0],
    'head':  lambda v,e: _rneseq(v,"first")[0],
    'last':  lambda v,e: _rneseq(v,"last")[-1],
    'tl':    lambda v,e: _rneseq(v,"tl")[1:],
    'tail':  lambda v,e: _rneseq(v,"tl")[1:],
    'tlr':   lambda v,e: _rneseq(v,"tlr")[:-1],
    'front': lambda v,e: _rneseq(v,"tlr")[:-1],
    'init':  lambda v,e: _rneseq(v,"tlr")[:-1],
    'pick':  lambda v,e: (lambda a,b: (
        _rseq(b,"pick")[a-1] if a > 0 else _rseq(b,"pick")[a] if a < 0 else (_ for _ in ()).throw(FPError("error: pick 0"))
    ))(*_rpair(v,"pick")),
    'apndl': lambda v,e: (lambda a,b: [a]+_rseq(b,"apndl"))(*_rpair(v,"apndl")),
    'apndr': lambda v,e: (lambda a,b: _rseq(a,"apndr")+[b])(*_rpair(v,"apndr")),
    'distl': lambda v,e: (lambda a,b: [[a,x] for x in _rseq(b,"distl")])(*_rpair(v,"distl")),
    'distr': lambda v,e: (lambda a,b: [[x,b] for x in _rseq(a,"distr")])(*_rpair(v,"distr")),
    'reverse': lambda v,e: _rseq(v,"reverse")[::-1],
    'rotl':  lambda v,e: (lambda s: s[1:]+[s[0]])(_rneseq(v,"rotl")),
    'rotr':  lambda v,e: (lambda s: [s[-1]]+s[:-1])(_rneseq(v,"rotr")),
    'trans': lambda v,e: _do_trans(v),
    'concat': lambda v,e: _do_concat(v),
    'pair':  lambda v,e: _do_pair(v),
    'split': lambda v,e: (lambda s: [s[:len(s)//2], s[len(s)//2:]])(_rseq(v,"split")),
    'iota':  lambda v,e: list(range(1, v+1)) if isinstance(v,int) and v>=0 else (_ for _ in ()).throw(FPError("error: iota expects non-negative integer")),
    'atom':  lambda v,e: is_atom(v) and v is not BOTTOM,
    'null':  lambda v,e: is_sequence(v) and len(v)==0,
    'length': lambda v,e: len(_rseq(v,"length")),
    'eq':    lambda v,e: fp_equal(*_rpair(v,"eq")),
    'not':   lambda v,e: not v if isinstance(v,bool) else (_ for _ in ()).throw(FPError("error: not expects boolean")),
    'and':   lambda v,e: (lambda a,b: a and b)(*_rpair(v,"and")) if all(isinstance(x,bool) for x in _rpair(v,"and")) else (_ for _ in ()).throw(FPError("error: and expects boolean pair")),
    'or':    lambda v,e: (lambda a,b: a or b)(*_rpair(v,"or")) if all(isinstance(x,bool) for x in _rpair(v,"or")) else (_ for _ in ()).throw(FPError("error: or expects boolean pair")),
    '+': _arith('+', lambda a,b: a+b),
    '-': _arith('-', lambda a,b: a-b),
    '*': _arith('*', lambda a,b: a*b),
    '/': _arith('/', lambda a,b: a//b if isinstance(a,int) and isinstance(b,int) else a/b),
    'mod': _arith('mod', lambda a,b: a%b),
    '=': _cmp('=', lambda a,b: a==b),
    'lt': _cmp('lt', lambda a,b: a<b),
    'gt': _cmp('gt', lambda a,b: a>b),
    'le': _cmp('le', lambda a,b: a<=b),
    'ge': _cmp('ge', lambda a,b: a>=b),
    'ne': _cmp('ne', lambda a,b: a!=b),
    'id': lambda v,e: v,
    'out': lambda v,e: (e._output_buffer.append(fp_format(v)), v)[1],
}

def _do_trans(v):
    s = _rseq(v, "trans")
    if not s: return []
    for r in s:
        if not is_sequence(r): raise FPError("error: trans expects sequence of sequences")
    ml = min(len(r) for r in s)
    return [[r[i] for r in s] for i in range(ml)]

def _do_concat(v):
    s = _rseq(v, "concat"); r = []
    for item in s:
        if is_sequence(item): r.extend(item)
    return r

def _do_pair(v):
    s = _rseq(v, "pair"); r = []; i = 0
    while i < len(s):
        if i+1 < len(s): r.append([s[i],s[i+1]]); i += 2
        else: r.append([s[i]]); i += 1
    return r

# ============================================================
# AST FORMATTER
# ============================================================

def format_ast(nd):
    cn = type(nd).__name__
    if cn == 'ASTAtom':      return fp_format(nd.value)
    if cn == 'ASTFuncRef':   return nd.name
    if cn == 'ASTCompose':
        # Wrap children that would parse differently without parens
        left_s = _fmt_compose_child(nd.left)
        right_s = _fmt_compose_child(nd.right)
        return "{} @ {}".format(left_s, right_s)
    if cn == 'ASTConstruct': return "[{}]".format(" ".join(format_ast(f) for f in nd.funcs))
    if cn == 'ASTApplyAll':  return "&{}".format(format_ast(nd.func))
    if cn == 'ASTInsertR':
        s = "!{}".format(format_ast(nd.func))
        if nd.seed is not None: s += "({})".format(format_ast(nd.seed))
        return s
    if cn == 'ASTInsertL':
        s = "\\{}".format(format_ast(nd.func))
        if nd.seed is not None: s += "({})".format(format_ast(nd.seed))
        return s
    if cn == 'ASTCondition':
        return "({} -> {} ; {})".format(format_ast(nd.pred), format_ast(nd.then_f), format_ast(nd.else_f))
    if cn == 'ASTConstant':  return "%{}".format(format_ast(nd.value))
    if cn == 'ASTApply':     return "{} : {}".format(format_ast(nd.func), format_ast(nd.arg))
    if cn == 'ASTSequence':  return "<{}>".format(" ".join(format_ast(e) for e in nd.elements))
    return "?"

def _fmt_compose_child(nd):
    """Format a child of ASTCompose, adding parens if needed for correct re-parsing."""
    cn = type(nd).__name__
    # These node types, when serialized bare, would capture the @ or parse differently
    if cn in ('ASTInsertR', 'ASTInsertL', 'ASTApplyAll', 'ASTCondition'):
        return "({})".format(format_ast(nd))
    return format_ast(nd)

# ============================================================
# REPL LOGIC
# ============================================================

HELP_TEXT = """
Berkeley FP -- 21st Century Edition

Operators:
  f : x          apply f to x
  f @ g          composition
  &f             apply-to-all
  !f             right-insert (foldr)
  !f(z)          right-insert with seed z
  \\f             left-insert (foldl)
  \\f(z)          left-insert with seed z
  [f g h]        construction
  p -> f ; g     condition
  %x             constant function (~ also accepted)
  n              selector (1, 2, 3, ...)

Commands:
  )help          this message
  )fns           list user-defined functions
  )pfn [name]    show function definition(s) (all if no name)
  )undef name    delete function
  )prims         list primitive functions
  )reset         clear all user functions
  )save name     save workspace to file
  )load          load workspace from file
  )ex            show examples

Values:
  <1 2 3>        sequence
  <>             empty sequence
  T F            boolean
  42  3.14       numbers
  abc            symbol

Definitions:
  {name body}    define a function

Precedence (high to low):
  () [] <>       grouping, construction, sequence
  ! \\ %          insert, constant (tight: binds next atom)
  &              apply-to-all (captures through @)
  @              composition (left-associative)
  -> ;           condition
  :              application (right-associative, lowest)"""

EXAMPLES_TEXT = """\
-- Basic application
first : <1 2 3>                       --> 1
tl : <1 2 3>                          --> <2 3>
2 : <10 20 30>                        --> 20
2 : <a b c>                           --> b

-- Composition
(first @ tl) : <1 2 3>               --> 2

-- Apply-to-all
&first : <<1 2> <3 4>>               --> <1 3>
&(+ @ [id %1]) : <1 2 3>             --> <2 3 4>

-- Insert (fold)
!+ : <1 2 3 4>                        --> 10
!+ @ iota : 10                        --> 55
\\- : <1 2 3>                          --> -4
!+(0) : <>                            --> 0

-- Construction
[first last] : <1 2 3>               --> <1 3>

-- Condition
(null -> %0 ; length) : <>            --> 0

-- User definitions
{double * @ [id %2]}
double : 7                             --> 14
{eq0 eq @ [id %0]}
{sub1 - @ [id %1]}
{fact eq0 -> %1 ; (* @ [id fact @ sub1])}
fact : 6                               --> 720

-- Inner product
{ip (!+) @ (&*) @ trans}
ip : <<1 2 3> <4 5 6>>                --> 32"""

def process_input(line, evaluator):
    stripped = line.strip()
    if not stripped:
        return ("", False)
    if stripped.startswith(')'):
        return _handle_cmd(stripped, evaluator)
    if stripped.startswith('{'):
        return _handle_def(stripped, evaluator)
    try:
        evaluator.reset_steps()
        evaluator._output_buffer = []
        ast = parse(stripped)
        result = evaluator.eval_expr(ast)
        parts = list(evaluator._output_buffer)
        cn = type(result).__name__
        if cn in ('ASTCompose','ASTConstruct','ASTApplyAll',
                  'ASTInsertR','ASTInsertL','ASTCondition','ASTConstant'):
            parts.append("<function: {}>".format(format_ast(result)))
        else:
            parts.append(fp_format(result))
        return ("\n".join(parts), False)
    except FPError as e:
        return (str(e), True)
    except Exception as e:
        return ("error: {}".format(e), True)

def _handle_cmd(line, ev):
    parts = line[1:].strip().split()
    if not parts: return ("error: empty command", True)
    cmd = parts[0].lower()
    if cmd == 'help': return (HELP_TEXT, False)
    if cmd in ('ex','examples'): return (EXAMPLES_TEXT, False)
    if cmd == 'fns':
        if ev.user_funcs: return (" ".join(sorted(ev.user_funcs.keys())), False)
        return ("(no user functions defined)", False)
    if cmd == 'pfn':
        names = parts[1:] if len(parts) >= 2 else sorted(ev.user_funcs.keys())
        if not names: return ("(no user functions defined)", False)
        r = []
        for nm in names:
            if nm in ev.user_funcs: r.append("{{{} {}}}".format(nm, format_ast(ev.user_funcs[nm])))
            else: r.append("{}: not defined".format(nm))
        return ("\n".join(r), False)
    if cmd in ('undef','delete'):
        if len(parts) < 2: return ("error: )undef requires function name(s)", True)
        r = []
        for nm in parts[1:]:
            if nm in ev.user_funcs: del ev.user_funcs[nm]; r.append("{}: deleted".format(nm))
            else: r.append("{}: not defined".format(nm))
        return ("\n".join(r), False)
    if cmd == 'prims': return (" ".join(sorted(set(PRIMITIVES.keys()))), False)
    if cmd == 'reset': ev.user_funcs.clear(); return ("all user functions cleared", False)
    if cmd == 'save':
        if not ev.user_funcs:
            return ("error: no functions to save", True)
        name = parts[1] if len(parts) >= 2 else "workspace"
        st.session_state._save_request = name
        return ("ready to save '{}' ({} functions)".format(name, len(ev.user_funcs)), False)
    if cmd == 'load':
        st.session_state._load_request = True
        return ("use the file uploader in the sidebar to load a .fp script", False)
    return ("error: unknown command '{}'".format(cmd), True)

def serialize_workspace(ev):
    """Serialize all user-defined functions to FP script text."""
    lines = ["-- Berkeley FP workspace", ""]
    for name in sorted(ev.user_funcs.keys()):
        lines.append("{{{} {}}}".format(name, format_ast(ev.user_funcs[name])))
    lines.append("")
    return "\n".join(lines)

def load_script(text, ev):
    """Load definitions from script text. Returns (count_loaded, errors)."""
    loaded = 0
    errors = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('--'):
            continue
        if line.startswith('{'):
            try:
                ast = parse(line)
                cn = type(ast).__name__
                if cn == 'ASTDef':
                    ev.user_funcs[ast.name] = ast.body
                    loaded += 1
                else:
                    errors.append("not a definition: {}".format(line))
            except FPError as e:
                errors.append("{}: {}".format(line[:30], e))
        else:
            errors.append("skipped: {}".format(line[:40]))
    return loaded, errors

def _handle_def(line, ev):
    try:
        ast = parse(line)
        cn = type(ast).__name__
        if cn == 'ASTDef':
            ev.user_funcs[ast.name] = ast.body
            return ("{{{}}}".format(ast.name), False)
        return ("error: invalid definition", True)
    except FPError as e:
        return (str(e), True)

# ============================================================
# STREAMLIT UI -- Chat-style with terminal aesthetics
# ============================================================

BANNER = "Berkeley FP, v.21.0 (21st Century Edition)"
ATTRIBUTION = "reimplemented in Python + Streamlit by Myung Ho Kim, 2026"

TERMINAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ===== GLOBAL RESET ===== */
:root {
    --bg: #161616;
    --bg2: #1c1c1c;
    --fg: #b8b8b0;
    --fg-bright: #e0e0dc;
    --fg-dim: #606060;
    --accent: #d4a04a;
    --error: #c25450;
    --border: #2a2a2a;
    --font: 'IBM Plex Mono', 'Menlo', 'Consolas', monospace;
}
*, *::before, *::after {
    font-family: var(--font) !important;
}
.stApp, .main, .block-container {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
}
header[data-testid="stHeader"] {
    background-color: var(--bg) !important;
}
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"],
#MainMenu, footer, .stDeployButton {
    display: none !important;
}

/* ===== BANNER ===== */
.fp-banner {
    color: var(--accent);
    font-size: 13px;
    font-weight: 600;
    padding: 12px 0 8px 0;
    letter-spacing: 0.3px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.fp-attrib {
    color: #908a7e;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.fp-workspace {
    color: green;
    font-size: 13px;
    font-weight: 600;
    padding: 12px 0 8px 0;
    letter-spacing: 0.3px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}

/* ===== TERMINAL OUTPUT ===== */
.fp-terminal {
    font-size: 13px;
    line-height: 1.7;
    padding: 4px 0 16px 0;
    white-space: pre-wrap;
    word-break: break-all;
}
.fp-terminal .fp-prompt { color: var(--fg-bright); }
.fp-terminal .fp-result { color: var(--fg); }
.fp-terminal .fp-error  { color: var(--error); }
.fp-terminal .fp-def    { color: var(--accent); }
.fp-terminal .fp-system { color: var(--fg); }

/* ===== CHAT INPUT — kill every white wrapper ===== */
[data-testid="stBottom"],
[data-testid="stBottom"] > *,
[data-testid="stChatInput"],
[class*="stChatFloating"],
[class*="stChatInput"],
.stChatInput,
div:has(> [data-testid="stChatInput"]) {
    background-color: var(--bg) !important;
    background: var(--bg) !important;
    border: none !important;
}
[data-testid="stChatInput"] {
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea,
.stChatInput textarea {
    font-size: 13px !important;
    background-color: #0e0e0e !important;
    color: var(--fg-bright) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    caret-color: var(--accent) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--fg-dim) !important;
}
[data-testid="stChatInput"] button,
.stChatInput button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: none !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #111 !important;
}
section[data-testid="stSidebar"] .fp-banner {
    font-size: 12px;
    padding: 8px 0 4px 0;
}
/* Sidebar text input */
section[data-testid="stSidebar"] input {
    font-size: 12px !important;
    background-color: #0a0a0a !important;
    color: var(--fg-bright) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}
/* Sidebar download button */
section[data-testid="stSidebar"] .stDownloadButton button {
    font-size: 12px !important;
    background-color: var(--bg2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}
section[data-testid="stSidebar"] .stDownloadButton button:hover {
    background-color: #282820 !important;
    border-color: var(--accent) !important;
}
/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: var(--fg-dim) !important;
    font-size: 11px !important;
}
/* Sidebar labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: var(--fg-dim) !important;
    font-size: 11px !important;
}
/* Sidebar file uploader — dark theme with readable text */
section[data-testid="stSidebar"] [data-testid="stFileUploader"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
    background-color: #0e0e0e !important;
    color: var(--fg) !important;
    border-color: var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: var(--fg) !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: var(--bg2) !important;
    color: var(--fg-bright) !important;
    border: 1px solid #444 !important;
    border-radius: 0 !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background-color: #303030 !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}
</style>
"""

def _md_escape(text):
    """Escape text for safe display inside st.markdown with unsafe_allow_html=True.
    HTML-escape first, then double backslashes so markdown doesn't eat them."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\\", "\\\\")
    return text

def main():
    st.set_page_config(page_title="Berkeley FP", page_icon="fp", layout="centered")
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    # Version check: force reset on code change
    if st.session_state.get("app_version") != APP_VERSION:
        st.session_state.evaluator = Evaluator()
        st.session_state.messages = []
        st.session_state.app_version = APP_VERSION

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = Evaluator()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    ev = st.session_state.evaluator

    # ---- Sidebar: Save / Load ----
    with st.sidebar:
        for _ in range(2):
            st.markdown("")
        st.markdown('<div class="fp-workspace">workspace</div>', unsafe_allow_html=True)

        # Save section
        if ev.user_funcs:
            save_name = st.text_input(
                "filename",
                value=st.session_state.get("_save_request", "workspace"),
                key="save_name_input",
                placeholder="workspace",
                label_visibility="collapsed",
            )
            script_text = serialize_workspace(ev)
            fname = save_name.strip() if save_name.strip() else "workspace"
            if not fname.endswith(".fp"):
                fname += ".fp"
            st.download_button(
                label="save {} ({})".format(fname, len(ev.user_funcs)),
                data=script_text,
                file_name=fname,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("no functions to save")

        st.markdown("---")

        # Load section
        uploaded = st.file_uploader(
            "load .fp script",
            type=["fp", "txt"],
            key="fp_loader_{}".format(st.session_state.get("_load_counter", 0)),
            label_visibility="collapsed",
        )
        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="replace")
            count, errors = load_script(text, ev)
            report = "loaded {} function(s) from '{}'".format(count, uploaded.name)
            if errors:
                report += "\n" + "\n".join(errors)
            st.session_state.messages.append({"role": "user", "content": ")load " + uploaded.name})
            st.session_state.messages.append({
                "role": "assistant",
                "content": report,
                "kind": "system"
            })
            # Bump loader key to allow re-uploading the same file
            st.session_state["_load_counter"] = st.session_state.get("_load_counter", 0) + 1
            # Clear save request flag
            if "_save_request" in st.session_state:
                del st.session_state["_save_request"]
            st.rerun()

    # ---- Main area ----
    # Banner
    st.markdown('<div class="fp-banner">{}<br><span class="fp-attrib">{}</span></div>'.format(BANNER, ATTRIBUTION), unsafe_allow_html=True)

    # Display message history as terminal output (no chat bubbles/avatars)
    if st.session_state.messages:
        lines = []
        prev_role = None
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            kind = msg.get("kind", "result")
            escaped = _md_escape(content)

            # Add blank line between answer/prompt pairs for visual separation
            if role == "user" and prev_role == "assistant":
                lines.append("")

            if role == "user":
                lines.append('<span class="fp-prompt">&gt; {}</span>'.format(escaped))
            elif kind == "error":
                lines.append('<span class="fp-error">{}</span>'.format(escaped))
            elif kind == "def":
                lines.append('<span class="fp-def">{}</span>'.format(escaped))
            elif kind == "system":
                lines.append('<span class="fp-system">{}</span>'.format(escaped))
            else:
                lines.append('<span class="fp-result">{}</span>'.format(escaped))

            prev_role = role

        html = "\n".join(lines)
        st.markdown(
            '<div class="fp-terminal">{}</div>'.format(html),
            unsafe_allow_html=True,
        )

    # Chat input (pinned to bottom)
    user_input = st.chat_input("Enter FP expression or )help")

    if user_input:
        stripped = user_input.strip()
        if not stripped:
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": stripped})

        # Process
        output, is_error = process_input(stripped, ev)

        # Determine output kind
        if is_error:
            kind = "error"
        elif stripped.startswith('{'):
            kind = "def"
        elif stripped.startswith(')'):
            kind = "system"
        else:
            kind = "result"

        if output:
            st.session_state.messages.append({"role": "assistant", "content": output, "kind": kind})

        st.rerun()

if __name__ == "__main__":
    main()
