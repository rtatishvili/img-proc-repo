export PYTHONPATH="$PYTHONPATH:../:../../common"
echo $PYTHONPATH
for file in $( ls test_*.py )
  do
    echo "======================================================================"	
    echo $file
    python $file
  done

