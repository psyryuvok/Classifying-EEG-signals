%Cod pentru desc?rcarea semnalelor de pe physionet
%Descarci toate fisierele din baza de date EEG Motor Movement/Imagery
str = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"];
for i=1:109
 if i<10
     for j=1:14
     url = sprintf('https://physionet.org/physiobank/database/eegmmidb/S00%d/S00%dR%s.edf', i,i,str(j));
     filename = sprintf('S00%dR%s.edf',i,str(j));
     outfilename = websave(filename,url);
     
     url2 = sprintf('https://physionet.org/physiobank/database/eegmmidb/S00%d/S00%dR%s.edf.event', i,i,str(j));
     filename2 = sprintf('S00%dR%s.edf.event',i,str(j));
     outfilename = websave(filename2,url2);
     end
 elseif i<100
     for j=1:14
    url = sprintf('https://physionet.org/physiobank/database/eegmmidb/S0%d/S0%dR%s.edf', i,i,str(j));
    filename = sprintf('S0%dR%s.edf',i,str(j));
    outfilename = websave(filename,url);
    
    url2 = sprintf('https://physionet.org/physiobank/database/eegmmidb/S0%d/S0%dR%s.edf.event', i,i,str(j));
    filename2 = sprintf('S0%dR%s.edf.event',i,str(j));
    outfilename = websave(filename2,url2);
    
    end
 else
     for j=1:14
     url = sprintf('https://physionet.org/physiobank/database/eegmmidb/S%d/S%dR%s.edf', i,i,str(j));
     filename = sprintf('S%dR%s.edf',i,str(j));
     outfilename = websave(filename,url);
     
     url2 = sprintf('https://physionet.org/physiobank/database/eegmmidb/S%d/S%dR%s.edf.event', i,i,str(j));
     filename2 = sprintf('S%dR%s.edf.event',i,str(j));
     outfilename = websave(filename2,url2);
     end
 end
