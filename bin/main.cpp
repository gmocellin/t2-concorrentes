#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "processamento.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    
    Mat *imagem;
    imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(!image.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    
    processamento *P = NULL;
    
    if (strcmp (argv[1], "thread") == 0)
        P = new paralelo(imagem);
    else if (strcmp (argv[1], "normal") == 0)
        P = new sequencial(imagem);
    
    char *nome = (char*) malloc ((strlen(argv[2])+1)*sizeof(char));
    strncpy(nome,argv[2],(strlen(argv[2])-4));
    strcat(nome,".ppm");
    
    Mat *nova_imagem = P->processando();
    imwrite(nome,nova_imagem);
    
    delete P;
    delete imagem;
    delete nova_imagem;
    free(nome);
    
    return 0;
}