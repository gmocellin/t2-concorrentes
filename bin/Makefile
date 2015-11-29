OPENCVFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect
LINKERFLAGS=-I/usr/local/include -Wall -g -O3 -std=c++11 -fopenmp
LINKERS=main.o
ZIP=t2-concorrentes.zip Makefile *.cpp
REMOVE=*.o new*.jpg t2-concorrentes
HOSTFILE=hostfile

all: $(LINKERS)
	@mpic++ $(LINKERS) $(LINKERFLAGS) -o t2-concorrentes -fopenmp $(OPENCVFLAGS) 
	@nvcc t3-concorrentes.cu -o t3-concorrentes ${ARGS}

main.o:
	@mpic++ -c $(LINKERFLAGS) main.cpp -fopenmp $(OPENCVFLAGS)

run_mpi:
	@mpirun -np 8 -hostfile $(HOSTFILE) ./t2-concorrentes thread ${ARGS}

run:
	@./t2-concorrentes normal ${ARGS}

run_cuda:
	@./t3-concorrentes ${ARGS}

tests-normal:
	@./tests-normal.sh

tests-thread:
	@./tests-thread.sh

tests-cuda:
	@./tests-cuda.sh

zip:
	@zip -r $(ZIP)

clean:
	@rm -rf $(REMOVE)
