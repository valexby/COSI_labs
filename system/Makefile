CC=g++
CFLAGS=-c -Wall -Wpedantic -g -O0
LDFLAGS=`pkg-config --libs opencv`
SOURCES=main.cpp
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=lab

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o
	rm $(EXECUTABLE)
