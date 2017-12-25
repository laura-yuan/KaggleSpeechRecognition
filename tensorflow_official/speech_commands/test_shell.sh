#!/bin/sh

a=0

while [ $a -lt 10 ]
do
   echo $a
   a=`expr $a + 1`
done


for i in 1 2 6 8 9 
do
    j=0
    while [ $j -lt $i ]
    do
      echo file_${i}_${j}
      j=`expr $j + 1` 
    done
done
