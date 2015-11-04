OPENCVFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect
LINKERFLAGS=-c -g -O3
COMPILEFLAGS=-g -O3 -o
LINKERS=main.o
ZIP=t2-concorrentes.zip Makefile *.cpp
REMOVE=*.o *new.jpg t2-concorrentes

all: $(LINKERS)
	@g++ $(OPENCVFLAGS) $(LINKERS) $(COMPILEFLAGS) t2-concorrentes

main.o:
	@g++ $(OPENCVFLAGS) $(LINKERFLAGS) main.cpp

run:
	@./t2-concorrentes ${ARGS}

tests:
	@./tests.sh

zip:
	@zip -r $(ZIP)

clean:
	@rm -rf $(REMOVE)
