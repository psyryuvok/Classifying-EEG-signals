# Classifying-EEG-signals

Au fost folosit bazele de date:	BCI Competition III-Data set IVa ‹motor imagery, small training sets› si Physionet- EEG Motor Movement/Imagery.<br/>
S-au folosit 2 metode:<br/>
Una care presupunea cross-corelare canalelor combinată cu diverși clasificatori. Aceasta a fost folosita pe baza de date de la BCI Competition.<br/>
Iar una care presupunea folosirea unei retele convolutionale care a fost folosita pe baza de date Physionet datorita dimensunii acesteia.<br/>
Initial s-a fost folosit Matlab, s-a trecut la Python pentru avea acces la librariile de machine learning.<br/>
Datele de pe BCI Competition III au fost descarcate manual. Acestea au fost prelucracte cu CCLRc care apeleaza edfread si eventread pentru deschiderea fisierelor, godfilc pentru filtrare, CodAdvancedc pentru prelucrarea lor.<br/>
Datele de pe Physionet au fost descarcate cu DownloadData.m si prelucracte cu PrelucrarePhysionetData.<br/>
Pentru clasificarea datelor de pe BCI Competition III, s-a folosit ChooseBestParameters care a apelat MultipleClassifiers. In urma clasificarii s-a obtinut un dictionar cu acuratetile obtiune.<br/>
Pentru clasificarea datelor de pe Physionet s-a folosit ReteaPython sau ReteaMatlab. 
