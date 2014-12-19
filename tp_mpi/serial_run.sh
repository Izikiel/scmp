for (( i = 20; i <= 1500; i += 10 )); do
	declare avg=0
	val=0
	for (( j = 0; j < 1000; j++ )); do
		val=$(./tp_mpi S 0.1 100 $i)
		if [[ -z "$val" ]]; then
			let j--
		else
			let avg+=val 
		fi
	done
	let avg=$avg/$j
	echo $avg
done

