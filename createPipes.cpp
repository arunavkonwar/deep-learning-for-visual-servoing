#include<stdio.h> 
#include<fcntl.h>
#include<stdlib.h>
#include <sys/stat.h>
main(){
  int file1,file2, file3;
  int fd;
  char str[256];
  char temp[4]="how";
  char temp1[4];
  file1 = mkfifo("/dev/shm/fifo_server",0777); 
  if(file1<0) {
    printf("Unable to create a fifo server\n");
    exit(-1);
  }

  file2 = mkfifo("/dev/shm/fifo_client",0777);

  if(file2<0) {
    printf("Unable to create a fifo client");
    exit(-1);
  }

  file3 = mkfifo("/dev/shm/fifo",0777);

  if(file1<0) {
    printf("Unable to create a fifo global");
    exit(-1);
  }
  printf("fifos server and child created successfuly\n");
}
