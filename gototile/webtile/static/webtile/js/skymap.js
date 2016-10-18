function onDateChanged() {
	var els = document.getElementsByClassName("datevaluespan");
	if (document.getElementById("id_date").value == 'specify') {	
		for (var i = 0; i < els.length; i++) {			
			els[i].style.display = 'inline';
		}
		document.getElementById("id_datevalue").focus();
	} else {
		for (var i = 0; i < els.length; i++) {			
			els[i].style.display = 'none';
		}
	}
}



document.addEventListener("DOMContentLoaded", function(event) {
	
	dateSelect = document.getElementById("id_date");
	dateSelect.addEventListener("change", onDateChanged, false);
	if (dateSelect.value != 'specify') {
		var els = document.getElementsByClassName("datevaluespan");
		for (var i = 0; i < els.length; i++) {
			els[i].style.display = 'none';
		}
	}
	
});
