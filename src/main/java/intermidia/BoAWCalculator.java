/*Teste de mod para commit*/

package intermidia;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.quantised.QuantisedLocalFeature;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.feature.local.keypoints.KeypointLocation;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import com.opencsv.CSVReader;

import TVSSUnits.Shot;
import TVSSUnits.ShotList;

public class BoAWCalculator 
{
	private static int k = 15;
	private static int clusteringSteps = 50;
	
    public static void main( String[] args ) throws Exception 
    {    	
    	if(args[2] != null)
    	{
    		k = Integer.parseInt(args[2]);
    	}
    	//Read SIFT features from CSV file.
    	CSVReader featureReader = new CSVReader(new FileReader(args[0]), ' ');
		String [] line;
		ShotList shotList = new ShotList();
		int lastShot = -1;
		
		
		
		//Build shot list with SIFT keypoints
		while ((line = featureReader.readNext()) != null) 
		{
			int currentShot = Byte.parseByte(line[0]);
			//It must be a while because there can be shots without keypoints
			while(currentShot != lastShot)
			{
				shotList.addShot(new Shot());
				lastShot++;
			}
			
			int fvSize = line.length - 1;
			byte fv[] = new byte[fvSize];
			
			for(int i = 0; i < fvSize; i++)
			{
				fv[i] = Byte.parseByte(line[i + 1]);
			}
			shotList.getLastShot().addSiftKeypoint(new Keypoint(0, 0, 0, 0, fv));
		}
		featureReader.close();
		
		//Build SIFT map per shot
		Map<Shot, LocalFeatureList<Keypoint>> videoKeypoints = new HashMap<Shot, LocalFeatureList<Keypoint>>();
		for(Shot shot: shotList.getList())
		{
			videoKeypoints.put(shot, shot.getSiftKeypointList());			
		}
		
		//Compute feature dictionary
		DataSource<byte []> kmeansDataSource = new LocalFeatureListDataSource<Keypoint, byte[]>(videoKeypoints);
		ByteKMeans clusterer = ByteKMeans.createExact(k, clusteringSteps);
		//$centroids have size $k, and each vector have 128 bytes
		System.out.println("Clustering SIFT Keypoints into "+ k + " visual words.");
		ByteCentroidsResult centroids = clusterer.cluster(kmeansDataSource);
		
		
		//Create the assigner, it is capable of assigning a feature vector to a cluster (to a centroid)
		HardAssigner<byte[], float[], IntFloatPair> hardAssigner = centroids.defaultHardAssigner();
		
		
    	//Compute features of each shot
		int shotn = 0;
		FileWriter bovwWriter = new FileWriter(args[1]);
		for(Shot shot: shotList.getList())
		{
			System.out.println("Processing shot " + shotn);
			//Print shot number
			bovwWriter.write(shotn++ + " ");
			
			//Variable quantisedFeatures assign a cluster label between [1..k] to each feature vector from the list 
			List<QuantisedLocalFeature<KeypointLocation>> quantisedFeatures = BagOfVisualWords.computeQuantisedFeatures(hardAssigner, shot.getSiftKeypointList());

			//Create the visual word ocurrence histogram
			SparseIntFV features = BagOfVisualWords.extractFeatureFromQuantised(quantisedFeatures, k);
			for(int i = 0; i < features.length(); i++)
			{
				bovwWriter.write(features.getVector().get(i) + " ");
			}
			bovwWriter.write("\n");			
		}
		bovwWriter.close();
    }
}

