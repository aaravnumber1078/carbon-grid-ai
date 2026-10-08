with open('tests/test_phase_e.py', 'rb') as f:
    content = f.read()

# Fix the indentation issue - second occurrence
pattern = b'or "concurrent" in result.error.lower()\r\n        \r\nt1.join(timeout=2.0)\r\n\n    def test_sequential_requests_work'
replacement = b'or "concurrent" in result.error.lower()\r\nt1.join(timeout=2.0)\r\n\r\n    def test_sequential_requests_work'

if pattern in content:
    content = content.replace(pattern, b'or "concurrent" in result.error.lower()\r\nt1.join(timeout=2.0)\r\n\r\n    def test_sequential_requests_work')
    with open('tests/test_phase_e.py', 'wb') as f:
        f.write(content)
    print('Fixed!')
else:
    print('Pattern not found')
    idx = content.find(b't1.join(timeout=2.0)')
    idx = content.find(b't1.join(timeout=2.0)', idx + 1)
    print(repr(content[idx:idx+200]))