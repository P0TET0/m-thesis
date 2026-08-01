score = [74, 85, 69, 77, 81]

high = [s for s in score if s>=80]

print("テストの点は", score, "です。")
print("80点以上は", high, "です。")
print("80点以上の人数は", len(high), "です。")