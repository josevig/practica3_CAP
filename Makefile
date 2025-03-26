CC=gcc

prueba: matrices_OpenMP.c 
	$(CC) -fopenmp  matrices_OpenMP.c -o  matrices_OpenMP.exe

clean: 
	rm *.exe
