OPENCVFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect
LINKERFLAGS=-c -g -O3
COMPILEFLAGS=-g -O3 -o
LINKERS=main.o
ZIP=t2-concorrentes.zip Makefile *.cpp
REMOVE=*.o new*.jpg t2-concorrentes

all: $(LINKERS)
	@mpic++ $(OPENCVFLAGS) $(LINKERS) $(COMPILEFLAGS) t2-concorrentes -fopenmp

main.o:
	@mpic++ $(OPENCVFLAGS) $(LINKERFLAGS) main.cpp -fopenmp

run_mpi:
	@mpirun -np 8 ./t2-concorrentes thread ${ARGS}

run:
	@./t2-concorrentes normal ${ARGS}

tests-normal:
	@./tests-normal.sh

tests-thread:
	@./tests-thread.sh

zip:
	@zip -r $(ZIP)

clean:
	@rm -rf $(REMOVE)
