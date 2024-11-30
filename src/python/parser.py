from dataclasses import dataclass
from typing import List  # Ensure List is imported
from .tokenizer import Token, TokenType


@dataclass
class TreeNode:
    pass


@dataclass
class BinOp(TreeNode):
    op: str
    left: TreeNode
    right: TreeNode


@dataclass
class Int(TreeNode):
    value: int


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.next_token_index: int = 0

    def eat(self, expected_token_type: TokenType) -> Token:
        token = self.tokens[self.next_token_index]
        self.next_token_index += 1
        if token.type != expected_token_type:
            raise RuntimeError(f"Expected {expected_token_type}, got {token.type}")
        return token

    def peek(self) -> TokenType:
        return self.tokens[self.next_token_index].type

    def parse(self) -> TreeNode:
        return self.parse_expression()

    def parse_expression(self) -> TreeNode:
        node = self.parse_term()

        while self.peek() in {TokenType.PLUS, TokenType.MINUS}:
            op = "+" if self.peek() == TokenType.PLUS else "-"
            self.eat(TokenType.PLUS if op == "+" else TokenType.MINUS)
            node = BinOp(op, node, self.parse_term())

        return node

    def parse_term(self) -> TreeNode:
        node = self.parse_factor()

        while self.peek() in {TokenType.MUL, TokenType.DIV}:
            op = "*" if self.peek() == TokenType.MUL else "/"
            self.eat(TokenType.MUL if op == "*" else TokenType.DIV)
            node = BinOp(op, node, self.parse_factor())

        return node

    def parse_factor(self) -> TreeNode:
        if self.peek() == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.parse_expression()
            self.eat(TokenType.RPAREN)
            return node
        else:
            token = self.eat(TokenType.INT)
            return Int(token.value)
