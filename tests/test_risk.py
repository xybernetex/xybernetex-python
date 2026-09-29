"""The OpenClaw plugin's test/risk.test.js, translated: the same commands must
get the same tiers in both ports."""
import unittest

from xybernetex.core.risk import classify_shell_command, classify_tool_call, meets_risk_threshold


class RiskTest(unittest.TestCase):
    def expect_all(self, cases):
        for command, expected in cases:
            self.assertEqual(classify_shell_command(command), expected, command)

    def test_exec_commands_from_real_runs_get_the_right_tier(self):
        # Verbatim from the 2026-09-25 test sessions, where every one of these
        # was judged destructive because the tool is named exec.
        self.expect_all([
            ("python scratch/fib.py", "none"),
            ("python scratch\\primes.py", "none"),
            ('python -m unittest discover -s scratch -p "test_*.py" -v', "none"),
            ("Get-ChildItem -Force | Select-Object Mode, Length, Name | Format-Table -AutoSize", "none"),
            ("Get-ChildItem -Path . -Filter *.md -Recurse -File | Select-Object -ExpandProperty FullName", "none"),
            ('if (Test-Path .\\supervisor-test.txt) { Write-Output "FOUND: supervisor-test.txt" } else { Write-Output "MISSING" }; '
             "$pyc = Get-ChildItem -Path .\\scratch -Recurse -Directory -Filter __pycache__", "none"),
            ('Remove-Item -Path .\\supervisor-test.txt -Force; if (Test-Path .\\supervisor-test.txt) { Write-Output "ERROR: still exists" } '
             'else { Write-Output "DELETED: supervisor-test.txt" }', "destructive"),
        ])

    def test_destructive_shell_commands_bash_and_powershell(self):
        self.expect_all([
            ("rm -rf build", "destructive"),
            ("sudo rm /etc/hosts", "destructive"),
            ("ls *.log | xargs rm", "destructive"),
            ("find . -name '*.tmp' -delete", "destructive"),
            ("find . -name '*.tmp' -exec rm {} \\;", "destructive"),
            ("del /q C:\\temp\\*", "destructive"),
            ("Get-ChildItem .\\logs | Remove-Item -Recurse", "destructive"),
            ("Clear-Content .\\notes.txt", "destructive"),
            ("git reset --hard HEAD~3", "destructive"),
            ("git clean -fdx", "destructive"),
            ("git push --force origin main", "destructive"),
            ("git branch -D old-feature", "destructive"),
            ("docker system prune -af", "destructive"),
            ("kubectl delete pod web-1", "destructive"),
            ("terraform destroy -auto-approve", "destructive"),
            ("npx wrangler kv key delete version --binding WEIGHTS", "destructive"),
            ('psql -c "DROP TABLE users"', "destructive"),
            ('sqlite3 app.db "DELETE FROM sessions"', "destructive"),
            ("npm unpublish my-pkg@1.0.0", "destructive"),
        ])

    def test_sensitive_shell_commands_reach_outside_the_machine(self):
        self.expect_all([
            ("git push origin main", "sensitive"),
            ("npm publish", "sensitive"),
            ("yarn publish", "sensitive"),
            ("npx wrangler deploy", "sensitive"),
            ("gh pr create --fill", "sensitive"),
            ("docker push registry.example.com/app:1", "sensitive"),
            ("terraform apply", "sensitive"),
            ("scp build.zip user@host:/srv", "sensitive"),
            ("curl -X POST https://api.example.com/items -d '{}'", "sensitive"),
            ('Invoke-RestMethod -Uri https://api.example.com -Method "POST" -Body $json', "sensitive"),
            ("Send-MailMessage -To a@b.c -Subject hi", "sensitive"),
            ("taskkill /IM node.exe /F", "sensitive"),
        ])

    def test_npm_publish_dry_run_is_a_simulation_not_a_publish(self):
        self.expect_all([
            ("npm publish --dry-run", "none"),
            ("npm publish --access public --dry-run", "none"),
            ("yarn publish --dry-run", "none"),
            ("pnpm publish --dry-run", "none"),
        ])

    def test_ordinary_work_stays_none_including_lookalikes(self):
        self.expect_all([
            ("git status", "none"),
            ('git commit -m "drop table support, delete from cache"', "none"),
            ("git log --oneline -5", "none"),
            ("npm install lodash", "none"),
            ("npm rm lodash", "none"),
            ("npm run build", "none"),
            ("pip install requests", "none"),
            ("curl https://example.com", "none"),
            ("kubectl get pods", "none"),
            ("Clear-Host", "none"),
            ('echo "rm -rf is dangerous"', "none"),
            ("Format-Table -AutoSize", "none"),
            ("", "none"),
        ])

    def test_a_chain_is_as_risky_as_its_worst_part(self):
        self.assertEqual(classify_shell_command("npm test && git push"), "sensitive")
        self.assertEqual(classify_shell_command("git push; rm -rf dist"), "destructive")

    def test_non_shell_tools(self):
        self.assertEqual(classify_tool_call("exec", {"command": "python x.py"}), "none")
        self.assertEqual(classify_tool_call("exec", {"command": "rm -rf /"}), "destructive")
        self.assertEqual(classify_tool_call("read", {"file_path": "a.txt"}), "none")
        self.assertEqual(classify_tool_call("write", {"file_path": "notes.md", "content": "x"}), "none")
        self.assertEqual(classify_tool_call("write", {"file_path": "C:\\Users\\me\\.ssh\\config", "content": "x"}), "sensitive")
        self.assertEqual(classify_tool_call("edit", {"path": "app/.env.production"}), "sensitive")
        self.assertEqual(classify_tool_call("apply_patch", {"input": "*** Begin Patch\n*** Delete File: old.py\n*** End Patch"}), "destructive")
        self.assertEqual(classify_tool_call("apply_patch", {"input": "*** Begin Patch\n*** Update File: src/a.js\n*** End Patch"}), "none")
        self.assertEqual(classify_tool_call("message", {"action": "send", "to": "x", "text": "hi"}), "sensitive")
        self.assertEqual(classify_tool_call("message", {"action": "read"}), "none")
        self.assertEqual(classify_tool_call("message", {"action": "delete", "id": "1"}), "destructive")
        self.assertEqual(classify_tool_call("cron", {"action": "list"}), "none")
        self.assertEqual(classify_tool_call("cron", {"action": "add"}), "sensitive")
        self.assertEqual(classify_tool_call("github_publish", {}), "sensitive")
        self.assertEqual(classify_tool_call("web_fetch", {"url": "https://x"}), "none")
        self.assertIsNone(classify_tool_call("some_mcp_tool", {"x": 1}))
        self.assertIsNone(classify_tool_call("tool_call", {"id": "web_fetch"}))

    def test_python_side_tool_names_from_the_sdk_adapter(self):
        # The names the OpenAI Agents SDK adapter gives its tools (scenarios/sdk_agent.py).
        self.assertEqual(classify_tool_call("run_command", {"command": "rm -rf build"}), "destructive")
        self.assertEqual(classify_tool_call("write_file", {"path": ".env"}), "sensitive")
        self.assertEqual(classify_tool_call("read_file", {"path": ".env"}), "none")
        self.assertEqual(classify_tool_call("list_files", {}), "none")

    def test_commands_behind_shell_keywords_are_classified(self):
        # `if ...; then trash build; else rm -rf build; fi`: `else rm` must not read as a command named "else".
        for command in ["if true; then rm -rf data; fi", "if [ -d x ]; then echo; else rm -rf data; fi",
                        "for d in a b; do rm -rf $d; done", "while true; do rm -rf data; done", "! rm -rf data",
                        "{ rm -rf data; }", "if rm -rf data; then echo gone; fi"]:
            self.assertEqual(classify_shell_command(command), "destructive", command)
        for command in ["echo then done", "grep -r else src", "if [ -f a ]; then cat a; fi"]:
            self.assertEqual(classify_shell_command(command), "none", command)

    def test_commands_hidden_in_substitutions_and_shell_strings_are_classified(self):
        # $(...) and backticks run inside double quotes and unquoted heredocs, and
        # bash -c / eval run their string; quoting hid all of these before 0.4.2.
        # Single quotes and quoted heredocs stay literal. Same cases as the plugin.
        for command, tier in [
            ("echo \"$(rm -rf data)\"", "destructive"),
            ("x=\"$(rm -rf data)\"", "destructive"),
            ("echo \"`rm -rf data`\"", "destructive"),
            ("echo `rm -rf data`", "destructive"),
            ("bash -c \"rm -rf data\"", "destructive"),
            ("sh -c 'rm -rf data'", "destructive"),
            ("bash -lc 'cd x && rm -rf data'", "destructive"),
            ("eval \"rm -rf data\"", "destructive"),
            ("sudo sh -c \"git reset --hard\"", "destructive"),
            ("cmd /c del /s x", "destructive"),
            ("pwsh -Command \"Remove-Item x -Recurse\"", "destructive"),
            ("bash -c \"echo \\\"$(rm -rf data)\\\"\"", "destructive"),
            ("cat <<EOF\n$(rm -rf data)\nEOF", "destructive"),
            ("cat <<EOF > n.txt\nit's `rm -rf data`\nEOF", "destructive"),
            ("bash -c \"git push origin main\"", "sensitive"),
            ("echo 'rm -rf data'", "none"),
            ("echo '$(rm -rf data)'", "none"),
            ("git commit -m \"remove the rm -rf step\"", "none"),
            ("cat <<'EOF'\n$(rm -rf data)\n`rm x`\nEOF", "destructive"),
            ("echo \"$((1+2))\"", "none"),
            ("bash -c \"echo hi\"", "none"),
            ("bash script.sh", "none"),
            ("python3 -c \"print(1)\"", "none"),
            ("cat <<'EOF' > README.md\nRun `rm -rf build` to clean.\nEOF\necho done", "none"),
            ("echo \"exit=$rc; removed: $([ ! -f x.js ] && echo yes || echo no)\"", "none"),
            ("cat <<-EOF\n\t$(rm -rf data)\n\tEOF", "destructive"),
            ("a=$(ls); echo `pwd`", "none"),
            ("echo \"$(echo \"$(git reset --hard)\")\"", "destructive"),
        ]:
            self.assertEqual(classify_shell_command(command), tier, command)

    def test_risk_thresholds_are_ordered_and_unknown_never_meets_one(self):
        self.assertTrue(meets_risk_threshold("destructive", "sensitive"))
        self.assertTrue(meets_risk_threshold("sensitive", "sensitive"))
        self.assertFalse(meets_risk_threshold("none", "sensitive"))
        self.assertFalse(meets_risk_threshold(None, "none"))
        with self.assertRaises(ValueError):
            meets_risk_threshold("none", "critical")


if __name__ == "__main__":
    unittest.main()
