package ml.classifier;

import java.io.File;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class Misc_code 
{
	// NOTE: You need headers for both class and features!
	public static void csvtoArff(String csv, String arff) throws IOException
	{
		// load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File(csv));
	    Instances data = loader.getDataSet();

	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(arff));
	    saver.writeBatch();
	}
	
    public static void writeClassifier(String path, Classifier clf) throws Exception
    {
    	SerializationHelper.write(path, clf);
    }
    
    public static Classifier loadClassifier(String path) throws Exception
    {
    	Classifier cls = (Classifier) SerializationHelper.read(path);
    	return cls;
    }
}
