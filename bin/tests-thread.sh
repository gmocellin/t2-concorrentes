#! /bin/bash

# for each matrix file
for test_file in $(ls images)
do

	total_time_single=0
	total_time_multi=0

	# ten times to have an average
	for((i=0 ;i<10;i++))
	do
		(/usr/bin/time -f "\n%e" make run_mpi ARGS=\\"images/$test_file\\") 2>> results/tmp-mpi-$test_file 1>> results/result-mpi-$test_file
		total_time_single=$(python -c "print ($total_time_single + $(tail -n 1 results/tmp-mpi-$test_file))")


	done

	echo images/$test_file
	echo Average MPI Time: $(python -c "print (\"%.3f\" % ($total_time_single/10))")
	echo 

	#rm results/tmp-$test_file
done
