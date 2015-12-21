/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package genutils;


/**
 *
 * @author Thanos
 */
public class RbmOptions {
    
    public int nclasses = 10;    
    public double eta = 0.1;
    public double momentum = 0.5; 
    public int maxepoch = 4;
    public int avglast = 0;
    public double penalty = 2e-4;
    public int batchsize = 500;
    public boolean verbose = true;
    public boolean anneal = false;    
    public int numhid = 500;
    public boolean debug = false;
    public boolean restart = true;
    public int createSnapshotEvery = 100;
    
}
