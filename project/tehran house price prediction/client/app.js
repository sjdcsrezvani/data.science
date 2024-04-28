function getroomsvalue() {
    var uirooms =document.getElementsByName('uirooms');
    for (var i in uirooms) {
        if (uirooms[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}

function getelevatorvalue() {
    var uielevator =document.getElementsByName('unielevator');
    for (var i in uielevator) {
        if (uielevator[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}

function getparkingvalue() {
    var uiparking =document.getElementsByName('uniparking');
    for (var i in uiparking) {
        if (uiparking[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}

function getwarehousesvalue() {
    var uiwarehouse =document.getElementsByName('uniwarehouse');
    for (var i in uiwarehouse) {
        if (uiwarehouse[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}



function onclickedestimateprice() {
    console.log('estimate price button clicked');
    var sqm = document.getElementById('uiSqm');
    var age = document.getElementById('uiyear');
    var rooms = getroomsvalue();
    var elevator = getelevatorvalue();
    var parking = getparkingvalue ();
    var warehouse = getwarehousesvalue();
    var location = document.getElementById('uilocations');
    var estprice = document.getElementById('uiestimatedprice');

    var url = 'http://127.0.0.1:5000/predict_home_price';

    $.post(url, {
        total_area: parseFloat(sqm.value),
        age: parseInt(age.value),
        rooms: rooms,
        elevator: elevator,
        parking: parking,
        warehouse: warehouse,
        location: location.value
    },function(data,status) {
        console.log(data.estimated_price);
        estprice.innerHTML = '<h2>' + data.estimated_price.toString() + ' milion toman</h2>';
        console.log(status);
    })
}

function onPageLoad() {
    console.log( "document loaded" );
    var url = "http://127.0.0.1:5000/get_location_names";
    $.get(url,function(data, status) {
        console.log("got response for get_location_names request");
        if(data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uilocations");
            $('#uilocations').empty();
            for(var i in locations) {
                var opt = new Option(locations[i]);
                $('#uilocations').append(opt);
            }
        }
    });
  }
  
  window.onload = onPageLoad;