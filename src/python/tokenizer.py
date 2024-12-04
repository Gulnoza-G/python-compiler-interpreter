from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Generator


class TokenType(StrEnum):
    INT = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()  # Left parenthesis
    RPAREN = auto()  # Right parenthesis
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any = None


class Tokenizer:
    def __init__(self, code: str) -> None:
        self.code = code
        self.ptr: int = 0

    def next_token(self) -> Token:
        while self.ptr < len(self.code) and self.code[self.ptr] == " ":
            self.ptr += 1

        if self.ptr == len(self.code):
            return Token(TokenType.EOF)

        char = self.code[self.ptr]
        self.ptr += 1
        if char == "+":
            return Token(TokenType.PLUS)
        elif char == "-":
            return Token(TokenType.MINUS)
        elif char == "*":
            return Token(TokenType.MUL)
        elif char == "/":
            return Token(TokenType.DIV)
        elif char == "(":
            return Token(TokenType.LPAREN)
        elif char == ")":
            return Token(TokenType.RPAREN)
        elif char.isdigit():
            num = char
            while self.ptr < len(self.code) and self.code[self.ptr].isdigit():
                num += self.code[self.ptr]
                self.ptr += 1
            return Token(TokenType.INT, int(num))
        else:
            raise RuntimeError(f"Can't tokenize {char!r}.")

    def __iter__(self) -> Generator[Token, None, None]:
        while (token := self.next_token()).type != TokenType.EOF:
            yield token
        yield token 
