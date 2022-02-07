for script in Assignment_01-02/*.py
do
    for file in Assignment_01-02/*/*.netlist
    do
        echo "Running $script $file"
        python $script $file
        echo ""
    done
    echo "----------------------------------------------------------------------------------\n"
done
