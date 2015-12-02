#! /bin/bash

# for each matrix file
for test_file in $(ls images)
do

	total_time_single=0
	total_time_multi=0

	# ten times to have an average
	for((i=0 ;i<10;i++))
	do
		(/usr/bin/time -f "\n%e" make run_cuda ARGS=\\"images/$test_file\\") 2>> results/tmp-cuda-1024-$test_file 1>> results/result-cuda-$test_file
		total_time_single=$(python -c "print ($total_time_single + $(tail -n 1 results/tmp-cuda-1024-$test_file))")

	done

	echo images/$test_file
	echo Average Cuda Time: $(python -c "print (\"%.3f\" % ($total_time_single/10))")
	echo 

	#rm results/tmp-$test_file
done
