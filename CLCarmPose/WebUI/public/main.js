// https://stackoverflow.com/questions/44391448/electron-require-is-not-defined
const fs = require('fs'); 
const { exec }  = require('child_process');

import { MeshSurfaceSampler } from '../ThridParty/MeshSurfaceSampler.js';

// global variables
let scene,renderer, camera, orbControls, raycaster, pointclouds;

const pointer = new THREE.Vector2();
const threshold = 0.1;
let intersection = null;

// buffer that will contain the setting for c++ app aka this is the script simple txt file
var newbuffer = '';

//global Controls
//var GlobalPCTraform;

var panel;

var applicationSettings = { Clouds:' ',Models:' ', select_x:'0.0', select_y:'0.0', select_z:'0.0'
                            ,scene_leaf_size: '0.05 ',scene_refrence_points_number:5
                             , model_down_sampling_ratio: '0.05', angle_dawnsampling:'10.0'
                          }

//var sceneslib = { 'Scene Default': 'models/scenes/scene0.pcd'
//};

// some global variable that deal with flipping scene issue
var fillped  = false;
var og_matrix = new THREE.Matrix4 ();

var modellib = {
    'Model Default': 'data/model/carm.pcd'
};

/************************************************************ Main *******************************************************************/

start();
update();

/*************************************************************************************************************************************/
// functions
function onPointerDown( event ) 
{

  pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
  pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = threshold;
  raycaster.setFromCamera( pointer, camera );
  const intersections = raycaster.intersectObjects( pointclouds );
  intersection = ( intersections.length ) > 0 ? intersections[ 0 ] : null;

   // clear old object
   var oldobject = scene.getObjectByName('selectionbbox');

   if(oldobject)
   {
          scene.remove(oldobject);
   }

  if(intersection !== null)
   {
     // console.log(intersection.point);
    applicationSettings.select_x = intersection.point.x;
    applicationSettings.select_y = intersection.point.y;
    applicationSettings.select_z = intersection.point.z;


    var bbmin = new THREE.Vector3(intersection.point.x-0.5, intersection.point.y-0.5, intersection.point.z-0.5);
    var bbmax = new THREE.Vector3(intersection.point.x+0.5, intersection.point.y+0.5, intersection.point.z+0.5);
    const box = new THREE.Box3(bbmin,bbmax);

    const helper = new THREE.Box3Helper( box, 0xffff00 );
    helper.name = "selectionbbox";
    scene.add(helper);
    // don't for get to remove this later
    document.removeEventListener( 'pointerdown', onPointerDown );
   } 

}

function panelForPCControl()
{
  panel = new dat.GUI( { width: 420} );

  // stackoverflow.com/questions/57328700/using-a-switch-statement-to-load-multiple-objects-in-three-js
  const folder0 = panel.addFolder( 'Data Library' );
  
    var scens = fs.readdirSync('..\\data\\scenes');

  folder0.add(applicationSettings, 'Clouds', scens).onChange(function(){ 
    loadScenePC();
 } );

 applicationSettings.FlipScene = function()
 {
   var scn = scene.getObjectByName('Pcloud');
   scn.rotateZ(-Math.PI);
   scn.rotateY(-Math.PI);
   fillped = !fillped;
 }

 folder0.add(applicationSettings,'FlipScene').name('Filp the scene');

 folder0.add(applicationSettings, 'Models', modellib).onChange(function(){ loadModelPC();
 } );

 const folder1 = panel.addFolder( 'Parameters' );

 applicationSettings.SelectPoint = function()
 {
    document.addEventListener( 'pointerdown', onPointerDown );
 }

 folder1.add(applicationSettings,'SelectPoint').name('Select Point');
 folder1.add(applicationSettings, 'select_x').name('x =').listen();
 folder1.add(applicationSettings, 'select_y').name('y =').listen();
 folder1.add(applicationSettings, 'select_z').name('z =').listen();

 //folder1.add(applicationSettings, 'scene_leaf_size').name('Tunning parameter 1 =').listen().onChange(function(value)
 //{  
  
 //} );

 folder1.add(applicationSettings, 'scene_refrence_points_number',1,50,1).name('Tunning parameter 2 =').listen().onChange(function(value)
 {  
  
 } );

 folder1.add(applicationSettings, 'model_down_sampling_ratio').name('Tunning parameter 3 =').listen().onChange(function(value)
 {  
  
 });

    folder1.add(applicationSettings, 'angle_dawnsampling').name('Tunning parameter 4 =').listen().onChange(function (value) {

    });

 const folder2 = panel.addFolder( 'Pose Estimation' );

 // here we are going to generate simple text filethat have some information 
 // this file will be used by c++ app via linux command
 applicationSettings.PoseEstimation = function()
 {
      newbuffer += (applicationSettings.Models + '\n');
     newbuffer += ('data/scenes/' + applicationSettings.Clouds + '\n');
      // newbuffer += (applicationSettings.scene_leaf_size + '\n');
      newbuffer += (applicationSettings.scene_refrence_points_number + '\n');
      newbuffer += (applicationSettings.model_down_sampling_ratio + '\n');
     newbuffer += (applicationSettings.angle_dawnsampling + '\n');
      var picked = new THREE.Vector3 (applicationSettings.select_x, applicationSettings.select_y, applicationSettings.select_z);
      // if the scene is visual flipped i need to flip back the point selected because in ppf application the model is not fillped x') compilicated shit i know 
      if(fillped)
      {
          // filp point 
          var oldobject = scene.getObjectByName('Pcloud');
          og_matrix.getInverse(oldobject.matrix);
          picked.applyMatrix4 (og_matrix);
          
      }
      newbuffer += ( picked.x + ' '  +  picked.y +
                    ' '  + picked.z + '\n');

      
        if (!fs.existsSync('../data/estimate.txt')) {
                      // file does not exist just create it
            fs.writeFile('../data/estimate.txt', newbuffer, (err) => {
              
                      });

        }
        else{

            // file exist delet it and create new one
            fs.unlink('../data/estimate.txt', (err) => {
              if (err) {
                console.error(err)
                return
              }
            })

            fs.writeFile('../data/estimate.txt', newbuffer, (err) => {
              
            });
        }
      
 
      newbuffer = '';

      // run the cpp linux command
      // aka run the ppf program from this cmd
      // https://www.digitalocean.com/community/tutorials/how-to-launch-child-processes-in-node-js

     //exec('C:\\Users\\Abdelhak\\workspace\\CLCarmPose\\x64\\Release\\CLCarmPose.exe', (error, stdout, stderr) => {
     //   if (error) {
     //     console.error(`error: ${error.message}`);
     //     return;
     //   }
      
     //   if (stderr) {
     //     console.error(`stderr: ${stderr}`);
         
     //   }
      
     //   console.log(`stdout:\n${stdout}`);
     //});
 }

 folder2.add(applicationSettings,'PoseEstimation').name('Estimate');

 // this function will read the out put of the cpp app and use the matrix to transform the model
 applicationSettings.ShowPose = function()
 {
    // get the object first
    var modelobject = scene.getObjectByName('Mcloud');
    modelobject.position.set( 0, 0, 0 );
    modelobject.rotation.x = 0;
    modelobject.rotation.y = 0;
    modelobject.rotation.z = 0;

    var loader = new THREE.FileLoader();


    const matrixpose = new THREE.Matrix4();
    //load a text file 
    loader.load(
        // resource URL
        '../data/pose.txt',
    
        // onLoad callback
        function ( data ) {
            // output the text to the console
            
            var subdata = data.split( ' ' );
            var values = [];
            for(var i = 0; i < subdata.length; i++)
            {
                values.push(parseFloat(subdata[i]));
            }

            matrixpose.set( values[0], values[1], values[2], values[3],
              values[4], values[5], values[6], values[7],
              values[8], values[9], values[10], values[11],
              values[12], values[13], values[14], values[15] );

            // modelobject.geometry.applyMatrix(matrixpose);
            // modelobject.geometry.verticesNeedUpdate = true;
            // console.log(modelobject);

            if(fillped)
            {

              matrixpose.set( values[0], values[1], values[2], values[3],
                -values[4], -values[5], -values[6], -values[7],
                -values[8], -values[9], -values[10], -values[11],
                values[12], values[13], values[14], values[15] );

            }
            // set the new pos of the object
            modelobject.applyMatrix4(matrixpose);
            modelobject.updateMatrix();
            //console.log( matrixpose )
        }
    );

 }


 folder2.add(applicationSettings,'ShowPose').name('Show Estimated Pose');
}



/***********************************************************************************************/


function loadScenePC() 
{
  const objLoader = new THREE.PCDLoader();
  

  // clear old object
  var oldobject = scene.getObjectByName('Pcloud');

  if(oldobject)
  {
         scene.remove(oldobject);
    }

    objLoader.load('../data/scenes/' + applicationSettings.Clouds, (root) => {
    
    const center = root.geometry.boundingSphere.center;
    console.log(center)
    const pmat = new THREE.PointsMaterial( { size: 0.001, vertexColors: true } );
    const pointscloud = new THREE.Points( root.geometry, pmat );
    pointscloud.name = 'Pcloud';
    pointclouds  = [pointscloud]; // this is stupid
    //pointscloud.rotateZ(-Math.PI);
    //pointscloud.rotateY(-Math.PI);
  //   root.geometry.translate(-center.x, -center.y, -center.z);
    orbControls.target.set( -center.x, -center.y, -center.z );
    orbControls.update();
    scene.add(pointscloud);
    
  });
}


function loadModelPC() 
{
  const objLoader = new THREE.STLLoader();
  
  // clear old object
  var oldobject = scene.getObjectByName('Mcloud');

  if(oldobject)
  {
         scene.remove(oldobject);
  }
    objLoader.load('../data/model/carm.stl', (root) => {
    
    const material = new THREE.MeshPhongMaterial( { color: 0x00dd00, specular: 0x111111, shininess: 200 ,  opacity: 0.8, transparent: true} );

    const mesh = new THREE.Mesh( root, material);
    mesh.geometry.computeVertexNormals();
    mesh.name = 'Mcloud';
    mesh.DefaultMatrixAutoUpdate = false;
    //mesh.updateMatrix();
    scene.add(mesh);   
  });
}


function start()
{

  // get the container, the whole canvas
  const container = document.getElementById( 'container' );
  /* const containerImage = document.getElementById( 'viewer' ); */

  // setup camera
  camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set( 0, 0, 0 );
  camera.up.set( 0, 1, 0 );

  scene = new THREE.Scene();
   /* sceneMedical = new THREE.Scene(); */
  // create scene elements

  scene.background = new THREE.Color( 0xa0a0a0 );
  scene.fog = new THREE.Fog( 0xa0abaf, 20, 500 );

  const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x424242 );
  hemiLight.position.set( 0, 200, 0 );
  scene.add( hemiLight );

  const dirLight = new THREE.DirectionalLight( 0xffffff );
  dirLight.position.set( - 3, 10, - 10 );
  dirLight.castShadow = true;
  scene.add( dirLight );

  // extras
      
  var size = 100;
  var divisions = 100;
  

  var gridHelper = new THREE.GridHelper( size, divisions);
  scene.add( gridHelper );

  var axesHelper = new THREE.AxesHelper( 100 );
  scene.add( axesHelper );

  // setup the renderer and attach it to canvas

  renderer = new THREE.WebGLRenderer( { antialias: true } );
  renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize( window.innerWidth, window.innerHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
 


  // set up orbit control
  orbControls = new THREE.OrbitControls( camera, renderer.domElement );
  orbControls.enablePan = true;
  orbControls.enableZoom = true;

  orbControls.target.set( 0, 0, 0 );
  orbControls.update();

  panelForPCControl();

  container.appendChild( renderer.domElement );
  /*containerImage.appendChild( renderMedical.domElement ); */
}



function update()
{
  requestAnimationFrame( update );
  renderer.render( scene, camera );
  /* renderMedical.render( sceneMedical, cameraMedical ); */
}
