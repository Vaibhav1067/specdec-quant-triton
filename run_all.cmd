@echo off
set PYTHONIOENCODING=utf-8
set PY=C:\venvs\specq\Scripts\python
%PY% bench_matmul.py > results\log_matmul.txt 2>&1
%PY% regs.py > results\log_regs.txt 2>&1
%PY% bench_attention.py > results\log_attn.txt 2>&1
%PY% bench_graphs.py > results\log_graphs.txt 2>&1
echo done > results\DONE.txt
