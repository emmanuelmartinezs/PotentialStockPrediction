function init() {
  // Grab a reference to the dropdown select element
  var selector = d3.select("#selDataset");

  // Use the list of sample names to populate the select options
  d3.json("samples.json").then((data) => {
    var sampleNames = data.names;

    sampleNames.forEach((sample) => {
      selector
        .append("option")
        .text(sample)
        .property("value", sample);
    });

    // Use the first sample from the list to build the initial plots
    var firstSample = sampleNames[0];
    buildCharts(firstSample);
    buildMetadata(firstSample);
  });
}

// Initialize the dashboard
init();

function optionChanged(newSample) {
  // Fetch new data each time a new sample is selected
  buildMetadata(newSample);
  buildCharts(newSample);
}

// Demographics Panel 
function buildMetadata(sample) {
  d3.json("samples.json").then((data) => {
    var metadata = data.metadata;
    // Filter the data for the object with the desired sample number
    var resultArray = metadata.filter(sampleObj => sampleObj.id == sample);
    var result = resultArray[0];
    // Use d3 to select the panel with id of `#sample-metadata`
    var PANEL = d3.select("#sample-metadata");

    // Use `.html("") to clear any existing metadata
    PANEL.html("");

    // Use `Object.entries` to add each key and value pair to the panel
    // Hint: Inside the loop, you will need to use d3 to append new
    // tags for each key-value in the metadata.
    Object.entries(result).forEach(([key, value]) => {
      PANEL.append("h6").text(`${key.toUpperCase()}: ${value}`);
    });
  });
}

// 1. Create the buildCharts function.
function buildCharts(sample) {
  // 2. Use d3.json to load and retrieve the samples.json file 
  d3.json("samples.json").then((data) => {
    // 3. Create a variable that holds the samples array. 
    var samplesArrays = data.samples;
    console.log(samplesArrays);
    // 4. Create a variable that filters the samples for the object with the desired sample number.
    var resultsArrays = samplesArrays.filter(sampleObj => sampleObj.id == sample);
    var metadataArray = data.metadata.filter(sampleObj => sampleObj.id == sample);
    //  5. Create a variable that holds the first sample in the array.
    var resultOne = resultsArrays[0];
    var metadataResult = metadataArray[0];
    var washFreq = parseFloat(metadataResult.wfreq);

    // 6. Create variables that hold the otu_ids, otu_labels, and sample_values.
    var otu_ids = resultOne.otu_ids;
    var otu_labels = resultOne.otu_labels;
    var sample_values = resultOne.sample_values;
    
    // 7. Create the yticks for the bar chart.
    // Hint: Get the the top 10 otu_ids and map them in descending order  
    //  so the otu_ids with the most bacteria are last. 

    var yticks = otu_ids.slice(0, 10).map(otuID => `OTU ${otuID}`).reverse();

    // 8. Create the trace for the bar chart. 
    var barData = [
      {
        y: yticks,
        x: sample_values.slice(0, 10).reverse(),
        text: otu_labels.slice(0, 10).reverse(),
        type: "bar",
        orientation: "h",
      }
    ];
    // 9. Create the layout for the bar chart. 
    var barLayout = {
      //barData,
      //ticks: yticks,
      //x: [sample_values],
      //y: [otu_labels],
      title: "Top 10 Bacteria Cultures Found",
      margin: { t: 30, l: 150 }
    };
    // 10. Use Plotly to plot the data with the layout. 
    Plotly.newPlot("bar", barData, barLayout)

//DELIVERABLE 2
    // 1. Create the trace for the bubble chart.
    var bubbleData = [
      {
        x: otu_ids,
        y: sample_values,
        text: otu_labels,
        mode:"markers",
        marker: {size: sample_values, color: otu_ids, colorscale: "Earth"},
        type: "bubble"
      }
    ];

    // 2. Create the layout for the bubble chart.
    var bubbleLayout = {
      title: "Bacteria Cultures Per Sample",
      xaxis: {title: {text: 'OTU ID'}}
    };

    // 3. Use Plotly to plot the data with the layout.
    Plotly.newPlot("bubble", bubbleData, bubbleLayout);

    //Extract belly button washing frequency
    //d3.json("samples.json").then(function(data){
      //wfreq = data.metadata.map(person => person.wfreq);
      //console.log(wfreq);
   //});

    // 3. Create a variable that holds the washing frequency.
    //intToFloat
    //function intToFloat(num, decPlaces) { return num.toFixed(decPlaces); }
    //alert(intToFloat(12, 1)); // returns 12.0
    //alert(intToFloat(12, 2)); // returns 12.00

    //var washFreq = parseFloat(metadata.wfreq);
    //data.metadata.map(person => person.wfreq);
      //console.log(wfreq);

//DELIVERABLE 3
    // 4. Create the trace for the gauge chart.
    var gaugeData = [
      {
        domain: { x: [0, 1], y: [0, 1] },
        value: washFreq,
        title: {text: "Belly Button Washing Frequency Scrubs Per Week", font: {size: 10}},
        type: "indicator",
        mode: "gauge+number",
        //title: {text: "Belly Button Washing Frequency", font: {size: 20}},

        gauge: {
          axis: {range: [null, 10]},
          bar: {color: "black"},
          steps: [
            {range: [0,2], color: "red"},
            {range: [2,4], color: "orange"},
            {range: [4,6], color: "yellow"},
            {range: [6,8], color: "lawngreen"},
            {range: [8,10], color: "green"}
          ],
        }
      }     
    ];

    // 5. Create the layout for the gauge chart.
    var gaugeLayout = { 
      width: 600,
      height: 500,
      margin: {t: 0, b: 0}};
    //};

    // 6. Use Plotly to plot the gauge data and layout.
    Plotly.newPlot("gauge", gaugeData, gaugeLayout);
  });
}
//what is the Obj fuction that  you see and why wasnt gauge a new function;
