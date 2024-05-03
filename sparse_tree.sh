#!/bin/bash

ANTLR_JAR="antlr4.jar"
GRAMMAR="Solidity"
FLAG="-tree"
AST=".tree"
FILEPATH=$1
OUTPUT=$2
START_RULE=$3

alias antlr4='java -jar antlr4.jar'

alias grun='java -classpath $ANTLR_JAR:ast_target/ org.antlr.v4.gui.TestRig'


mkdir -p ast_target/

java -jar $ANTLR_JAR $GRAMMAR.g4 -o src/

javac -classpath $ANTLR_JAR src/*.java -d ast_target/

for file in `ls $1 | grep -E "*\.sol"`;
do
    if [ -f "$1/$file" ];
    then
       	name=`echo $file | cut -d. -f1`
	grun "$GRAMMAR" "$START_RULE" "$FLAG" "$1/$file" > "$2/${name}$AST" &
    fi
done 
   
