"""Code runner tool — safely executes Python code snippets."""

import sys
import io
import traceback
from .base import BaseTool, ToolResult

# Max output lines to prevent flooding
MAX_OUTPUT_LINES = 20


class CodeRunnerTool(BaseTool):
    name = 'code_runner'
    description = 'Runs Python code and returns the output. Input: valid Python code.'

    def run(self, input_text: str) -> ToolResult:
        code = input_text.strip()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            exec(code, {'__builtins__': __builtins__})  # noqa: S102
            output = stdout_capture.getvalue()
            error  = stderr_capture.getvalue()
        except Exception:
            output = stdout_capture.getvalue()
            error  = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Limit output length
        lines = (output or error or '(no output)').strip().splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[:MAX_OUTPUT_LINES] + [f'... ({len(lines) - MAX_OUTPUT_LINES} more lines)']
        result_text = '\n'.join(lines)

        return ToolResult(
            tool_name=self.name,
            success=not error,
            output=result_text,
            error=error if error else '',
        )
