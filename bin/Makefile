OPENCVFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect
LINKERFLAGS=-c -g -O3
COMPILEFLAGS=-g -O3 -o

all: processamento.o main.o
	@g++ $(OPENCVFLAGS) processamento.o main.o $(COMPILEFLAGS) t2-concorrentes

processamento.o:
	@g++ $(OPENCVFLAGS) $(LINKERFLAGS) processamento.cpp

main.o:
	@g++ $(OPENCVFLAGS) $(LINKERFLAGS) main.cpp

run:
	@./t2-concorrentes ${ARGS}

tests:
	@./tests.sh

zip:
	@zip -r t2-concorrentes.zip Makefile *.cpp *.h

clean:
	@rm *.o t2-concorrentes
