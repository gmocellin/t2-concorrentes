#include <iostream>
#include <cstring>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    
    Mat imagem, nova_imagem;
    imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    
    imagem.copyTo(nova_imagem);

    if (strcmp (argv[1], "thread") == 0) {
        
    }
    else if (strcmp (argv[1], "normal") == 0) {
        int i,j, linha, coluna;
        for (i = 0; i < imagem.rows; i++) {
            for (j = 0; j < imagem.cols; j++) {
                float sum[3] = {0, 0, 0};
                for(linha = (i - 2); linha <= (i + 2); linha++){
                    for(coluna = (j - 2); coluna <= (j + 2); coluna++) {
                        if ((linha >= 0 && linha < imagem.rows) && (coluna >= 0 && coluna < imagem.cols)) {
                            sum[0] += imagem.at<Vec3b>(linha,coluna)[0];
                            sum[1] += imagem.at<Vec3b>(linha,coluna)[1];
                            sum[2] += imagem.at<Vec3b>(linha,coluna)[2];
                        }
                    }
                }
                nova_imagem.at<Vec3b>(i,j)[0] = sum[0]/25;
                nova_imagem.at<Vec3b>(i,j)[1] = sum[1]/25;
                nova_imagem.at<Vec3b>(i,j)[2] = sum[2]/25;
            }
        }
    }
    
    char *nome = (char*) malloc ((strlen(argv[2])+1)*sizeof(char));
    strncpy(nome,argv[2],(strlen(argv[2])-4));
    strcat(nome,"new.jpg");
    
    imshow("teste",nova_imagem);
    waitKey(0);
    imwrite(nome,nova_imagem);
    
    free(nome);
    
    return 0;
}