import ast
import astor
from yapf.yapflib.yapf_api import FormatCode 

def code_replace(processor_code: str, obj):
    # Parse the input code into an AST
    tree = ast.parse(processor_code)

    # Define a new function to replace the body of get_parameters
    new_body = ast.parse(f"def get_parameters(self):\n    return {astor.to_source(ast.Constant(obj))}").body[0].body

    # Visitor to find and replace the get_parameters method
    class ReplaceGetParameters(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'get_parameters':
                # Replace the body of the get_parameters method
                node.body = new_body
            return node

    # Transform the AST
    transformed_tree = ReplaceGetParameters().visit(tree)

    # Convert the AST back to source code
    new_code = astor.to_source(transformed_tree)

    # Make it pretty
    formatted_code, _ = FormatCode(new_code)
    return formatted_code

def test():
    with open("algo_test.py", "r") as f:
        test_code = f.read()
        print(code_replace(test_code, {"test": 1}))


if __name__ == "__main__":
    test()
