# MY COMPILER
Python is **interpreted** but also involves a compilation step. Here's how it works:

1. **Compilation to Bytecode:** When you run a Python script, the Python interpreter first compiles the code into an intermediate form called **bytecode**. This bytecode is a low-level representation of your code, which is not human-readable but is optimized for execution.
2. **Execution by Python Virtual Machine (PVM):** The bytecode is then executed by the **Python Virtual Machine (PVM)**, which is the actual interpreter that runs the instructions.

- Python is **not compiled to machine code** directly like C or C++.
- The compilation step is **internal** and invisible to the user; you typically interact with Python as an interpreted language.
- This makes Python **platform-independent** because the bytecode can run on any system with a compatible Python interpreter.

## Structure of My Project

My program has a linear structure with four main components:

1. **Tokenizer**: Converts source code into tokens.
2. **Parser**: Transforms tokens into an Abstract Syntax Tree (AST).
3. **Compiler**: Converts the AST into bytecode.
4. **Interpreter**: Executes the bytecode to produce results.

## 1. Tokenizer

The **Tokenizer** is the first component of my program that processes source code. Its job is to break the code into **tokens**—small pieces that are easier for the program to understand and analyze. Tokens are classified based on their role, like numbers, operators, or parentheses.

---

### Why Do We Need a Tokenizer?

The tokenizer simplifies the source code by breaking it into manageable pieces, which can then be used for further processing like parsing or compilation. Without this step, understanding and analyzing raw code would be much harder.

---

### Token Types

I’ve defined 8 types of tokens to represent the essential elements of arithmetic expressions:

1. **`INT`**: Represents integers (e.g., `8`, `21`).
2. **`PLUS`**: The `+` operator.
3. **`MINUS`**: The `` operator.
4. **`MUL`**: The `` operator.
5. **`DIV`**: The `/` operator.
6. **`LPAREN`**: The `(` left parenthesis.
7. **`RPAREN`**: The `)` right parenthesis.
8. **`EOF`**: Marks the end of the source code.

---

### Token Representation

Each token is represented using a `Token` class with two attributes:

1. **`type`**: Defines the category of the token (e.g., `PLUS` or `INT`).
I used **`StrEnum`** to create these categories, and `auto()` to generate their values automatically.
2. **`value`**: Stores the actual value of the token (e.g., `8` for an integer).

### Example Code:

```python
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Generator

class TokenType(StrEnum):
    INT = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: Any = None
```

---

### How Does the Tokenizer Work?

The **Tokenizer** scans the input code and generates a sequence of tokens.

### Input Examples:

1. **Input**: `"5 * 7 + (10 - 3)"`
    
    **Output**:
    
    `[Token(INT, 5), Token(MUL), Token(INT, 7), Token(PLUS), Token(LPAREN), Token(INT, 10), Token(MINUS), Token(INT, 3), Token(RPAREN), Token(EOF)]`
    
2. **Input**: `"12 / (6 + 2)"`
    
    **Output**:
    
    `[Token(INT, 12), Token(DIV), Token(LPAREN), Token(INT, 6), Token(PLUS), Token(INT, 2), Token(RPAREN), Token(EOF)]`
    

---

### Logic Behind the Tokenizer

1. **Initialization**:
    
    The `Tokenizer` class accepts a code string and uses a pointer (`ptr`) to track the current position in the code.
    
    ```python
    class Tokenizer:
        def __init__(self, code: str) -> None:
            self.code = code
            self.ptr: int = 0
    ```
    
2. **Generating Tokens**:
    
    The `next_token` method skips whitespace and checks the current character to determine its token type.
    
    ```python
    def next_token(self) -> Token:
        # Skip whitespace
        while self.ptr < len(self.code) and self.code[self.ptr] == " ":
            self.ptr += 1
    
        # End of input
        if self.ptr == len(self.code):
            return Token(TokenType.EOF)
    
        # Identify token
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
            raise RuntimeError(f"Cannot tokenize character {char!r}.")
    ```
    

---

### Making the Tokenizer Iterable

To simplify usage, I made the `Tokenizer` class iterable. This allows easy token extraction using a `for` loop.

### Code:

```python
class Tokenizer:
    # ...

    def __iter__(self) -> Generator[Token, None, None]:
        while (token := self.next_token()).type != TokenType.EOF:
            yield token
        yield token
```

---

### Example Usage

### Input Code:

```python
code = "8 * (5 + 3) - 4"
tokenizer = Tokenizer(code)

for token in tokenizer:
    print(f"{token.type}, {token.value}")
```

### Output:

```
INT, 8
MUL, None
LPAREN, None
INT, 5
PLUS, None
INT, 3
RPAREN, None
MINUS, None
INT, 4
EOF, None
```

---

### Why This Approach?

- **Clarity**: Breaks down complex code into small, understandable pieces.
- **Reusability**: Tokens can be reused for parsing or other operations.
- **Flexibility**: Easily expandable for more complex syntax.

This modular design ensures the tokenizer is robust and straightforward to use.

## 2. Parser

The **Parser** is the part of the program that takes a sequence of tokens (produced by the tokenizer) and ensures that they follow valid syntax. It also constructs a **tree-like representation** of the program, called an **Abstract Syntax Tree (AST)**, to represent operations and their relationships.

---

### **Why Do We Need a Parser?**

While the tokenizer identifies the building blocks (tokens) of the code, the parser ensures these tokens are arranged meaningfully. For example:

- Code like `3 + 5` makes sense, so the parser creates a tree to represent this operation.
- Invalid code like `3 3 3 +` doesn’t make sense, so the parser raises an error.

---

### **How the Parser Works**

1. **Creates an Abstract Syntax Tree (AST):**
    
    The parser translates the tokens into a structured tree where:
    
    - Operators (e.g., `+`, ``) are root nodes.
    - Operands (e.g., `3`, `5`) are leaf nodes.
2. **Ensures Syntax Validity:**
    
    The parser uses logic to enforce valid syntax, such as ensuring:
    
    - Numbers must follow operators.
    - Parentheses are matched.

---

### **Abstract Syntax Tree (AST)**

An AST represents the structure of the code without focusing on its exact syntax. For example:

- The code `3 + 5` and a hypothetical prefix notation `+ 3 5` both produce the same AST:
    
    ```
         +
       /   \
      3     5
    ```
    

---

### **Key Components**

### **TreeNode and Subclasses**

These classes represent the nodes of the AST:

- **`TreeNode`:** Base class for all nodes.
- **`BinOp`:** Represents binary operations (e.g., `+`, ``, ``, `/`).
    - `op`: The operator (`+`, ``, etc.).
    - `left` and `right`: The operands (other `TreeNode` objects).
- **`Int`:** Represents integer values in the AST.

```python
from dataclasses import dataclass

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
```

### **Parser Class**

The parser uses helper methods to process tokens and build the AST:

1. **`eat(expected_token_type)`**:
Ensures the next token matches the expected type and consumes it.
2. **`peek()`**:
Peeks at the next token without consuming it.

---

### **Parsing Logic**

### **Basic Parsing Example**

Parsing a single addition or subtraction:

```python
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
```

### **Parsing Expressions with Precedence**

To handle precedence (`*` and `/` before `+` and `-`), the parser uses recursive methods:

1. **`parse_expression()`**:
Handles `+` and ``.
2. **`parse_term()`**:
Handles `` and `/`.
3. **`parse_factor()`**:
Handles integers and parenthesized expressions.

```python
class Parser:
		# ...

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
```

---

### **Example Usage**

### **Input Tokens:**

```python
[Token(INT, 2), Token(PLUS), Token(LPAREN), Token(INT, 3), Token(MUL), Token(INT, 4), Token(RPAREN)]
```

### **Parsing Code:**

```python
from tokenizer import Tokenizer
from parser import Parser

code = "2 + (3 * 4)"
tokens = list(Tokenizer(code))
parser = Parser(tokens)
ast = parser.parse_expression()
print(ast)
```

### **Output AST:**

```
       +
     /   \
    2     *
         /   \
        3     4
```

---

### **Error Handling**

The parser ensures the input is valid:

- If an unexpected token appears, it raises an error:
    
    ```
    RuntimeError: Expected INT, got Token(type=<TokenType.PLUS>, value=None)
    ```
    

---

## Understanding the Stack and the Role of a Compiler

The **stack** and the **compiler** are central components in transforming code into executable instructions. Here's a simplified explanation of their roles and the processes involved.

---

### **Why Do We Need a Stack?**

The **stack** is a simple data structure used for evaluating operations in a sequence. It operates on a **Last In, First Out (LIFO)** principle, making it ideal for handling nested or sequential operations.

### Example: Evaluating `3 + 5`

1. **Initial State:** The stack is empty.
2. **Push Operands:** Add `3` and `5` to the stack.
3. **Apply Operation:** The `+` operator pops the two values (`3` and `5`), adds them, and pushes the result (`8`) back onto the stack.

This model allows complex operations to be broken down into simple, atomic steps.

---

### **What Does the Compiler Do?**

The **compiler** converts the **Abstract Syntax Tree (AST)** from the parser into a sequence of **bytecodes**. Bytecodes are simple, low-level instructions that the interpreter can execute.

### Example:

For the expression `3 + 5`, the AST is:

```
       +
     /   \
    3     5
```

The compiler transforms this into bytecodes:

```
PUSH 3
PUSH 5
ADD
```

These bytecodes instruct the interpreter:

1. `PUSH 3`: Add `3` to the stack.
2. `PUSH 5`: Add `5` to the stack.
3. `ADD`: Pop the top two values (`3` and `5`), add them, and push the result (`8`).

---

### **How the Stack Works**

The stack processes bytecodes by performing operations step-by-step. Let's break it down:

### **Expression:** `2 * (3 + 4)`

### **Steps:**

1. **AST:** The parser builds the tree:
    
    ```
          *
        /   \
       2     +
            /   \
           3     4
    ```
    
2. **Bytecodes (Generated by the Compiler):**
    
    ```
    PUSH 3
    PUSH 4
    ADD
    PUSH 2
    MUL
    ```
    
3. **Stack Execution (Interpreted Bytecode):**
    - `PUSH 3`: Stack = `[3]`
    - `PUSH 4`: Stack = `[3, 4]`
    - `ADD`: Pop `3` and `4`, push `7`. Stack = `[7]`
    - `PUSH 2`: Stack = `[7, 2]`
    - `MUL`: Pop `7` and `2`, push `14`. Stack = `[14]`

### **Final Result:** `14`

---

## 3. Compiler

The **compiler** converts the Abstract Syntax Tree (AST) produced by the parser into **bytecode operations**, which are step-by-step instructions executed using a stack. Bytecodes are simple and help break down complex expressions into manageable operations.

---

### **Understanding Bytecode**

Bytecodes are instructions represented by:

1. **Type**: Defines the kind of operation (e.g., `PUSH`, `BINOP`).
2. **Value**: Additional data needed for the operation (e.g., numbers or operators).

```python
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Generator

class BytecodeType(StrEnum):
    BINOP = auto()  # Binary operation (e.g., +, -, *, /)
    PUSH = auto()   # Push a value onto the stack

@dataclass
class Bytecode:
    type: BytecodeType
    value: Any = None
```

---

### **Compiler Class**

The **Compiler** traverses the AST recursively and generates bytecodes for each node.

```python
class Compiler:
    def __init__(self, tree: TreeNode) -> None:
        self.tree = tree

    def compile(self) -> Generator[Bytecode, None, None]:
        # Generates bytecode instructions for the given AST
        yield from self._compile_node(self.tree)

    def _compile_node(self, node: TreeNode) -> Generator[Bytecode, None, None]:
        # Recursively compiles each node in the AST into bytecodes
        if isinstance(node, Int):
            # Push an integer value onto the stack
            yield Bytecode(BytecodeType.PUSH, node.value)
        elif isinstance(node, BinOp):
            # Compile the left and right operands
            yield from self._compile_node(node.left)
            yield from self._compile_node(node.right)
            # Add the operation as a bytecode
            yield Bytecode(BytecodeType.BINOP, node.op
```

---

### **Example: Simple Expression**

### **Input Code**

```python
code = "3 + 5"
```

1. **AST Representation**
    
    ```
        +
      /   \
     3     5
    ```
    
2. **Generated Bytecodes**
    
    ```
    [
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=3),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=5),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='+')
    ]
    ```
    

### **How it Works**

- `PUSH 3`: Push `3` onto the stack.
- `PUSH 5`: Push `5` onto the stack.
- `BINOP +`: Add the top two stack values (`3` and `5`) and push the result (`8`) back onto the stack.

---

### **Example: Nested Expression**

### **Input Code**

```python
code = "2 * (3 + 4)"

```

1. **AST Representation**
    
    ```
          *
        /   \
       2     +
            /   \
           3     4
    ```
    
2. **Generated Bytecodes**
    
    ```
    [
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=3),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=4),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='+'),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=2),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='*')
    ]
    ```
    

### **How it Works**

1. `PUSH 3`: Push `3` onto the stack.
2. `PUSH 4`: Push `4` onto the stack.
3. `BINOP +`: Add `3` and `4` to get `7`, push `7` onto the stack.
4. `PUSH 2`: Push `2` onto the stack.
5. `BINOP *`: Multiply `2` and `7` to get `14`, push `14` onto the stack.

---

### **Running the Compiler**

The following code processes input, compiles it into bytecode, and prints the result.

```python
if __name__ == "__main__":
    import sys
    from .tokenizer import Tokenizer
    from .parser import Parser

    # Input code passed as a command-line argument
    code = sys.argv[1]

    # Tokenize and parse the input
    tokens = list(Tokenizer(code))
    tree = Parser(tokens).parse()

    # Compile the AST into bytecodes
    compiler = Compiler(tree)
    bytecode = list(compiler.compile())

    # Output the generated bytecode
    print("Generated Bytecode:")
    for bc in bytecode:
        print(bc)
```

---

### **Execution Example**

### **Command**

```bash
python -m compiler "2 * (3 + 4)"
```

### **Output**

```
Generated Bytecode:
Bytecode(type=<BytecodeType.PUSH: 'push'>, value=3)
Bytecode(type=<BytecodeType.PUSH: 'push'>, value=4)
Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='+')
Bytecode(type=<BytecodeType.PUSH: 'push'>, value=2)
Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='*')
```

---

## 4. Interpreter

The **Interpreter** executes the bytecodes generated by the compiler using a **stack**. It evaluates expressions by performing operations in a step-by-step manner based on the bytecodes.

---

### **Why Do We Need an Interpreter?**

The interpreter reads bytecode instructions and:

1. **Manages the Stack:** Uses a stack to store intermediate results during computation.
2. **Executes Bytecode:** Processes each bytecode operation (`PUSH`, `BINOP`) to perform arithmetic or manipulate the stack.

---

### 1. **Stack Class**

The stack is a simple data structure that stores intermediate values. It operates on a **Last In, First Out (LIFO)** principle.

```python
class Stack:
    def __init__(self) -> None:
        self.stack: List[int] = []  # Initialize an empty stack.

    def push(self, item: int) -> None:
        # Push an integer onto the stack
        self.stack.append(item)

    def pop(self) -> int:
        # Pop the top value off the stack
        if not self.stack:
            raise RuntimeError("Cannot pop from an empty stack!")
        return self.stack.pop()

    def __repr__(self) -> str:
        # Return a string representation of the stack
        return f"Stack({self.stack})"

```

**Example Usage of Stack:**

```python
stack = Stack()
stack.push(5)
stack.push(3)
print(stack)  # Output: Stack([5, 3])
stack.pop()   # Removes 3
print(stack)  # Output: Stack([5])

```

---

### 2. **Interpreter Class**

The interpreter reads the bytecodes and uses the stack to compute the results.

```python
class Interpreter:
    def __init__(self, bytecode: List[Bytecode]) -> None:
        self.stack = Stack()       # Stack for intermediate results.
        self.bytecode = bytecode   # List of bytecode instructions.

    def interpret(self) -> None:
        """Interpret the bytecode and compute the result."""
        for bc in self.bytecode:
            if bc.type == BytecodeType.PUSH:
                # Push a value onto the stack.
                self.stack.push(bc.value)
            elif bc.type == BytecodeType.BINOP:
                # Perform a binary operation.
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
        print(self.stack)  # Display the final state of the stack.
```

---

### **Execution Process**

1. **Input Bytecode:**
The interpreter receives bytecodes like:
    
    ```
    [
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=3),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=5),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='+')
    ]
    ```
    
2. **Execution:**
    - `PUSH 3`: Push `3` onto the stack.
    - `PUSH 5`: Push `5` onto the stack.
    - `BINOP +`: Pop `3` and `5`, add them (`3 + 5 = 8`), and push `8` back onto the stack.
3. **Final State of the Stack:**
    
    ```
    Stack([8])
    ```
    

---

### **Main Execution Block**

This block ties together the tokenizer, parser, compiler, and interpreter to process the input code.

```python
if __name__ == "__main__":
    import sys
    from .tokenizer import Tokenizer
    from .parser import Parser
    from .compiler import Compiler

    # Input code provided as a command-line argument
    code = sys.argv[1]

    # Tokenize the input
    tokens = list(Tokenizer(code))

    # Parse the tokens into an AST
    tree = Parser(tokens).parse()

    # Compile the AST into bytecode
    bytecode = list(Compiler(tree).compile())

    # Interpret the bytecode to compute the result
    Interpreter(bytecode).interpret()
```

---

### **Example Execution**

### **Input Code:**

```bash
python -m interpreter "2 * (3 + 4)"
```

### **Process:**

1. **Tokenization:**
    
    ```
    [Token(INT, 2), Token(MUL), Token(LPAREN), Token(INT, 3), Token(PLUS), Token(INT, 4), Token(RPAREN)
    ```
    
2. **Parsing (AST):**
    
    ```
          *
        /   \
       2     +
            /   \
           3     4
    
    ```
    
3. **Compilation (Bytecode):**
    
    ```
    [
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=3),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=4),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='+'),
        Bytecode(type=<BytecodeType.PUSH: 'push'>, value=2),
        Bytecode(type=<BytecodeType.BINOP: 'binop'>, value='*')
    ]
    ```
    
4. **Interpretation (Stack Operations):**
    - `PUSH 3`: Stack = `[3]`
    - `PUSH 4`: Stack = `[3, 4]`
    - `BINOP +`: Stack = `[7]` (`3 + 4 = 7`)
    - `PUSH 2`: Stack = `[7, 2]`
    - `BINOP *`: Stack = `[14]` (`7 * 2 = 14`)

### **Output:**

```
Done!
Stack([14])
```
