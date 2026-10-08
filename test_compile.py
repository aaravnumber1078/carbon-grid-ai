code = "            \"predicted_prob\": float(prob),"
try:
    compile(code, "<string>", "exec")
    print("Compiles OK")
except SyntaxError as e:
    print("SyntaxError: " + str(e))
