# src/python/interpreter.py
from .compiler import Bytecode, BytecodeType
from typing import List


class Stack:
    def __init__(self) -> None:
        self.stack: List[int] = []

    def push(self, item: int) -> None:
        self.stack.append(item)

    def pop(self) -> int:
        return self.stack.pop()

    def __repr__(self) -> str:
        return f"Stack({self.stack})"


class Interpreter:
    def __init__(self, bytecode: List[Bytecode]) -> None:
        self.stack = Stack()
        self.bytecode = bytecode

    def interpret(self) -> None:
        for bc in self.bytecode:
            if bc.type == BytecodeType.PUSH:
                self.stack.push(bc.value)
            elif bc.type == BytecodeType.BINOP:
                b = self.stack.pop()
                a = self.stack.pop()
                if bc.value == "+":
                    self.stack.push(a + b)
                elif bc.value == "-":
                    self.stack.push(a - b)
                elif bc.value == "*":
                    self.stack.push(a * b)
                elif bc.value == "/":
                    self.stack.push(a // b)  # Integer division
                else:
                    raise RuntimeError(f"Unknown operator {bc.value}")
        print("Done!")
        print(self.stack)


if __name__ == "__main__":
    import sys
    from .tokenizer import Tokenizer
    from .parser import Parser
    from .compiler import Compiler

    code = sys.argv[1]
    tokens = list(Tokenizer(code))
    tree = Parser(tokens).parse()
    bytecode = list(Compiler(tree).compile())
    Interpreter(bytecode).interpret()
