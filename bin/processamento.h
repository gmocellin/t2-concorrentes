#ifndef PROCESSAMENTO_H
#define PROCESSAMENTO_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class processamento {
protected:
    Mat *matriz, *R, *G, *B;
    
    void setRGB();
    void setMatriz(Mat *matriz);
    
    void smooth(Mat *matriz,int flag);
public:
    /* construtores */
    processamento(Mat *matriz);
    /* destrutores */
    virtual ~processamento();
    
    Mat *getR();
    Mat *getG();
    Mat *getB();
    Mat *getMatriz();
    
    virtual Mat processando();
};

class paralelo : public processamento {
public:
    /* construtores */
    paralelo(Mat *matriz);
    /* destrutores */
    ~paralelo();
    
    Mat processando();
};

class sequencial : public processamento {
public:
    /* construtores */
    sequencial(Mat *matriz);
    /* destrutores */
    ~sequencial();
    
    Mat processando();
};

#endif