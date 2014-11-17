echo "task_1_1" > timeit_results.txt
python -m timeit "from main import task_1_1; task_1_1()" >> timeit_results.txt
echo "task_1" >> timeit_results.txt
python -m timeit "from main import task_1; task_1()" >> timeit_results.txt
echo "task_1_2" >> timeit_results.txt
python -m timeit "from main import task_1_2; task_1_2()" >> timeit_results.txt
echo "task_2" >> timeit_results.txt
python -m timeit "from main import task_2; task_2()" >> timeit_results.txt
echo "task_1_3" >> timeit_results.txt
python -m timeit "from main import task_1_3; task_1_3()" >> timeit_results.txt
echo "task_3" >> timeit_results.txt
python -m timeit "from main import task_3; task_3()" >> timeit_results.txt