# t2-concorrentes
T2 de Concorrentes :)

- Como executar? (a partir do makefile criado)

Para executar o algoritmo do projeto 3:
	    
	1. Execute o comando 'make compile-cuda' para compilar o programa.
    
	2. Execute o comando 'make run\_cuda ARGS="argumento"' para rodar a versão cuda do programa.
  		- Em "argumento" coloque o caminho para a imagem e o nome dela. 
    	
	3. O comando 'make clean' irá apagar o executável.
    
	4. Para executar a bateria de testes basta usar os comandos 'make tests-cuda'.


	Caso queira executar os algoritmos do projeto 2:

	1. Execute o comando 'make' (sem aspas) para compilar o programa.

	2. Execute o comando 'make run ARGS="argumentos"' (sem aspas) para rodar o programa sequencialmente.
  		- Em 'argumentos' você coloca o nome do arquivo de imagem que será aberto, passado por argumento(argv) para o programa.
  
	3. Execute o comando 'make run\_mpi ARGS="argumentos"' (sem aspas) para rodar o programa paralelamente.
  		- Em 'argumentos' você coloca o nome do arquivo de imagem que será aberto, passado por argumento(argv) para o programa.
  
	4. Execute o comando 'make clean' (sem aspas) para apagar o arquivo executável e de compilação.
  
	5. Para executar a bateria de testes basta usar os comandos "make tests-normal" para o processamento sequencial e "make tests-thread" para o processamento paralelo.
