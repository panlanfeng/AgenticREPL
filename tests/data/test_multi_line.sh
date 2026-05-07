# test_multi_line.sh — mix of correct commands and typos for srun file mode
echo "=== Testing file execution ==="
ls
cat tests/data/test.csv
wc -l tests/data/test.csv
grep --nocolor Alice tests/data/test.csv
ll
pwd
echoo "hello world"
ls all inverse order
echo "=== Done ==="
