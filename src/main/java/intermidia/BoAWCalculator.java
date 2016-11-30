/*Teste de mod para commit*/

package intermidia;

import java.io.FileReader;
import java.io.FileWriter;

import org.openimaj.data.DataSource;
import org.openimaj.data.DoubleArrayBackedDataSource;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import com.opencsv.CSVReader;

import TVSSUnits.Shot;
import TVSSUnits.ShotList;

public class BoAWCalculator 
{
	private static int k = 30;
	private static int clusteringSteps = 50;
	
    public static void main( String[] args ) throws Exception 
    {    	
    	//Read MFCC features from CSV file.
    	CSVReader featureReader = new CSVReader(new FileReader(args[0]), ' ');
		String [] line;
		ShotList shotList = new ShotList();
		int lastShot = -1;	
		int mfccFVTotal = 0;
		int fvSize = 0;
		
		//Build shot list with MFCC keypoints
		while ((line = featureReader.readNext()) != null) 
		{
			int currentShot = Byte.parseByte(line[0]);
			//It must be a while because there can be shots without descriptors
			while(currentShot != lastShot)
			{
				shotList.addShot(new Shot());
				lastShot++;
			}
			
			fvSize = line.length - 1;
			DoubleFV fv = new DoubleFV(fvSize);
			
			for(int i = 0; i < fvSize; i++)
			{
				fv.setFromDouble(i, Double.parseDouble(line[i + 1]));
			}
			shotList.getLastShot().addMfccDescriptor(fv);
			mfccFVTotal++;
		}
		featureReader.close();
				
		//Build MFCC descriptor pool
		double[][] allMfccKeypoints = new double[mfccFVTotal][fvSize];
		int n = 0;		
		for(Shot shot: shotList.getList()) //Iterate shot list
		{
			for(int l = 0; l < shot.getMfccList().size(); l++) //Iterate mfcc list for a shot
			{
				for(int m = 0; m < fvSize; m++) //Iterate all positions of a FV
				{
					allMfccKeypoints[n][m] = shot.getMfccList().get(l).get(m);
				}
				n++;
			}
		}
		
		//Compute feature dictionary
		DataSource<double []> kmeansDataSource = new DoubleArrayBackedDataSource(allMfccKeypoints);
		DoubleKMeans clusterer = DoubleKMeans.createExact(k, clusteringSteps);
		//$centroids have size $k, and each vector have 13 double values
		System.out.println("Clustering MFCC Feature Vectors into "+ k + " aural words.");
		DoubleCentroidsResult centroids = clusterer.cluster(kmeansDataSource);
		
		//Create the assigner, it is capable of assigning a feature vector to a cluster (to a centroid)
		HardAssigner<double[], double[], IntDoublePair> hardAssigner = centroids.defaultHardAssigner();
		
    	//Compute features of each shot
		int shotn = 0;
		FileWriter boawWriter = new FileWriter(args[1]);
		for(Shot shot: shotList.getList())
		{
			System.out.println("Processing shot " + shotn);
			//Print shot number
			boawWriter.write(Integer.toString(shotn++));
			
			//Create and initialize aural word histogram for a shot			
			double[] mfccHistogram = new double[k];
			for(int i = 0; i < k; i++)
			{
				mfccHistogram[i] = 0;
			}
			
			//Assign each MFCC FV of a shot to an aural word
			for(DoubleFV mfccFV: shot.getMfccList())
			{
				//Increase the ocurrence of an aural word in the histogram
				mfccHistogram[hardAssigner.assign(mfccFV.values)]++;
			}
			
			//Set shot feature histogram for use in intershot distance

			Histogram featureHistogram = new Histogram(mfccHistogram);
			featureHistogram = new Histogram(featureHistogram.normaliseFV());
			shot.setFeatureWordHistogram(featureHistogram);
			
			for(int i = 0; i < k; i++)
			{
				boawWriter.write(" " + mfccHistogram[i]);
			}
			boawWriter.write("\n");			
		}
		boawWriter.close();
		
		//Print aural words to file
		FileWriter awWriter = new FileWriter(args[2]);
		for(int i = 0; i < centroids.numClusters(); i++)
		{
			for(int j = 0; j < centroids.numDimensions(); j++)
			{
				if(j < centroids.numDimensions() - 1)
				{
					awWriter.write(centroids.getCentroids()[i][j] + " ");
				}else
				{
					awWriter.write(centroids.getCentroids()[i][j] + "\n");
				}
			}
		}
		awWriter.close();
		
		
		//Print intershot distances
		for(int i = 0; i < (shotList.listSize() - 1); i++)
		{
			double intershotDist = shotList.getShot(i).getFeatureWordHistogram().compare(shotList.getShot(i + 1).getFeatureWordHistogram(), 
					DoubleFVComparison.COSINE_SIM);
			System.out.println("Sim " +  i + "/" + (i + 1) + ": " + intershotDist);
		}
    }
}

