
ScriptLoc=${PWD} #save script directory path as shell variable

cd LectureCodes

for i in *.py; do echo $i; python $i; done #run all python scripts in directory

grep “I HAVE WORKED” *

#NOTE: grep searches through

#files for matching strings (very useful command)

cd $ScriptLoc #return to script directory

for i in *.py; do echo $i; python $i; done #run all python scripts in directory
