import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instances;
import ml.classifier.IncrementalClassifier;
import weka.classifiers.evaluation.Evaluation;


public class main_weka_driver 
{
	/*
	 * Expects an ARFF file as first argument (class attribute is assumed
	 * to be the last attribute).
	 *
	 * @param args        the commandline arguments
	 * @throws Exception  if something goes wrong
	 */
	public static void main(String[] args) throws Exception 
	{
		UpdateableClassifier [] update_clf = IncrementalClassifier.init();
		Evaluation [] evaluation = new Evaluation[update_clf.length];
		Instances test_data = null;
		try
		{
			test_data = IncrementalClassifier.read_arff_file("C:\\Users\\Andrew\\Desktop\\ML_Module\\Weka_ML\\src\\ml\\classifier\\KDDTrain+.arff");
			IncrementalClassifier.train_incrementally(update_clf, evaluation, "C:\\Users\\Andrew\\Desktop\\ML_Module\\Weka_ML\\src\\ml\\classifier\\KDDTest+.arff");
		}
		catch(Exception e)
		{
			
		}
		
		// Get The Training Score of the Model
		for (int i = 0; i < update_clf.length; i++)
		{
			// Get Test Score
			evaluation[i].evaluateModel((Classifier) update_clf[i], test_data);
			System.out.println(evaluation[i].pctCorrect());
		}
	}
	
}
