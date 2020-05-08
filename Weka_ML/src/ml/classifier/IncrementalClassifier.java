package ml.classifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SGD;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.MultiClassClassifierUpdateable;
import weka.classifiers.trees.HoeffdingTree;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class IncrementalClassifier 
{
	private static final int CAPACITY = 10000;
	
	public static UpdateableClassifier [] init()
	{
		ArrayList<UpdateableClassifier> c = new ArrayList<UpdateableClassifier>();
		c.add(new NaiveBayesUpdateable());
		c.add(new HoeffdingTree());
		c.add(new NaiveBayesUpdateable());
		c.add(new HoeffdingTree());
		c.add(new IBk());
		c.add(new KStar());
		c.add(new LWL());
		c.add(new MultiClassClassifierUpdateable());
		c.add(new SGD());
		//c.add(new NaiveBayesMultinomialUpdateable());
		//c.add(new NaiveBayesMultinomialText());
		//c.add(new SGDText());
		return c.toArray(new UpdateableClassifier[c.size()]);
	}
	
	public static void train_incrementally(UpdateableClassifier [] clf, Evaluation [] evaluation, String filepath) throws Exception
	{
		ArffLoader load = new ArffLoader();
		load.setFile(new File(filepath));
		Instances train_data = load.getStructure();
		train_data.setClassIndex(train_data.numAttributes() - 1);
		Instance current = null;

		for (int i = 0; i < clf.length;i++)
		{
			Classifier x = (Classifier) clf[i];
			x.buildClassifier(train_data);
			evaluation[i] = new Evaluation(train_data);
		}
		
		while ((current = load.getNextInstance(train_data)) != null) 
		{
			for (UpdateableClassifier u: clf)
			{
				u.updateClassifier(current);
			}
		}
	}
	
	public static Instances read_arff_file(String filepath) throws IOException
	{
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(filepath));
		Instances data = loader.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	// In case I can't read it all in one shot...
	public static Instances read_arff_file_part(String filepath, int start) throws IOException
	{
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(filepath));
		Instances data = loader.getStructure();
		Instance current = null;
		
		// Do I need to shift my starting index?
		int counter = 0;
		while(counter != start)
		{
			current = loader.getNextInstance(data);
			if (current == null)
			{
				System.err.println("Out of bounds!");
				return null;
			}
			++counter;
		}
		
		// Read my specific amount
		counter = 0;
		while (counter != CAPACITY && (current = loader.getNextInstance(data)) != null)
		{
			data.add(current);
			++counter;
		}
		
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
}