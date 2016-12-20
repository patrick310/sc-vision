#define CAPTURE_WIDTH  500
#define CAPTURE_HEIGHT 375

    //Allocazione della matrice immagine per il frame catturato dalla telecamera
Mat frame(CAPTURE_HEIGHT, CAPTURE_WIDTH,CV_8UC3);

//Variabili di inizializzazione della Telecamera IDS uEye UI1220SE-C

HIDS hCam = 1; //Apre Camera con ID 1
int BITS_PER_PIXEL = 24;
int pWidth = CAPTURE_WIDTH;
int pHeight = CAPTURE_HEIGHT; 
SENSORINFO sensor_info;
CAMINFO camera_info;

//puntatori memoria
char* m_pcImageMemory;
int m_lMemoryId;

//Pulizia memoria da foto precedenti
if (hCam != 0){
    is_FreeImageMem (hCam,m_pcImageMemory,m_lMemoryId);
    is_ExitCamera(hCam);
}

//inizializzazione della telecamera 
int initcamera = is_InitCamera(&hCam, NULL);
if(initcamera != IS_SUCCESS)
{
    cout<<endl<<"Impossibile inizializzare la telecamera"<<endl;
    exit(-1);
}

// Acquisisce informazioni riguardanti la telecamera
int camerainfo = is_GetCameraInfo (hCam, &camera_info);
if(camerainfo != IS_SUCCESS)
{
    printf("Impossibile acquisire le informazioni della telecamera");
    exit(-1);
} 
// Acquisisce informazioni riguardanti il sensore della telecamera
int sensorinfo = is_GetSensorInfo (hCam, &sensor_info);
if(sensorinfo != IS_SUCCESS)
{
    printf("Impossibile acquisire le informazioni del sensore");
    exit(-1);
}

//Output informazioni camera/sensore
cout<<endl<<"<<< CARATTERISTICHE DELLA TELECAMERA COLLEGATA >>>"<<endl;
cout<<"Numero seriale: " << camera_info.SerNo << endl;
cout << "Produttore: " << camera_info.ID << endl;
cout << "Modello: " << sensor_info.strSensorName << endl;
cout << "Dimensioni massime per l'immagine: " << sensor_info.nMaxWidth << "x" << sensor_info.nMaxHeight << endl << endl;


//Imposta la modalità di colore BGR24 
int colormode = is_SetColorMode(hCam, IS_CM_BGR8_PACKED);
//int colormode = is_SetColorMode(hCam, IS_SET_CM_RGB24);
if(colormode != IS_SUCCESS)
{
    printf("Impossibile impostare il modo di colore");
    exit(-1);
}

//imposta dimensioni immagini che voglio catturare
int pXPos = (sensor_info.nMaxWidth);
int pYPos = (sensor_info.nMaxHeight);

//Inizializzazione Memoria camera
int rit = is_AllocImageMem (hCam,pXPos,pYPos, 24, &m_pcImageMemory, &m_lMemoryId);
if(rit != IS_SUCCESS)
{
    cout<<endl<<"IMPOSSIBILE INIZIALIZZARE LA MEMORIA"<<endl;
    system("PAUSE");
    exit(-1);
}
cout<<endl<<"Memoria inizializzata"<<endl;

//attivazione della locazione di memoria
int rat = is_SetImageMem (hCam, m_pcImageMemory, m_lMemoryId);
if(rat != IS_SUCCESS)
{
    cout<<endl<<"IMPOSSIBILE ATTIVARE LA MEMORIA"<<endl;
    system("PAUSE");
    exit(-1);
}
cout<<endl<<"Memoria Attivata"<<endl;

//impostazioni correzioni di colore
double strenght_factor = 1.0;
int colorcorrection = is_SetColorCorrection(hCam, IS_CCOR_ENABLE, &strenght_factor);

//impostazioni correzione del bianco
double pval = 1;
int whiteb = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &pval, 0);

//impostazione della correzione guadagno
double gval = 1;
int gains = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_GAIN, &gval, 0);


//inizio fase cattura immagine
int dummy;
char *pMem, *pLast;

//ciclo di ripetizione
for (int i=0;i<3;i++)
{

int sho = is_FreezeVideo(hCam, IS_WAIT);
if(sho != IS_SUCCESS)
{
    cout<<endl<<"IMPOSSIBILE ACQUISIRE DALLA TELECAMERA"<<endl;
    system("PAUSE");
    exit(-1);
}
if (sho == IS_SUCCESS){
    int m_Ret = is_GetActiveImageMem(hCam, &pLast, &dummy);
    int n_Ret = is_GetImageMem(hCam, (void**)&pLast);
   }

 IplImage* tmpImg = cvCreateImageHeader(cvSize (pXPos, pYPos), IPL_DEPTH_8U,3); 
 tmpImg->imageData = m_pcImageMemory;
 frame = cv::cvarrToMat(tmpImg);
 imshow("PROVA",frame);
 waitKey(0);

}
//chiusura e pulizia della telecamera
int en = is_ExitCamera(hCam);
 if (en == IS_SUCCESS){
    cout<<endl<<"Camera chiusa correttamente"<<endl;
 } 