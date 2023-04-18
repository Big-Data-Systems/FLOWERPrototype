var head = document.getElementsByTagName("head")[0];

var link = document.createElement("link");
link.type = "text/css";
link.rel = "stylesheet";
link.href = "../assets/css/bootstrap.min.css";
head.appendChild(link);

link = document.createElement("link");
link.type = "text/css";
link.rel = "stylesheet";
link.href = "../assets/css/main.css";
head.appendChild(link);

function goSamples() {
  // determine if it's an extension
  var isExtension = (location.pathname.split('/').slice(-2)[0].indexOf("extensions") >= 0);
  var isTS = (location.pathname.split('/').slice(-2)[0].indexOf("TS") > 0);

  // save the body for goViewSource() before we modify it
  window.bodyHTML = document.body.innerHTML;
  window.bodyHTML = window.bodyHTML.replace(/</g, "&lt;");
  window.bodyHTML = window.bodyHTML.replace(/>/g, "&gt;");

  // wrap the sample div and sidebar in a fluid container
  var container = document.createElement('div');
  container.className = "container-fluid";
  document.body.appendChild(container);

  // sample content
  var samplediv = document.getElementById('sample') || document.body.firstChild;
  samplediv.className = "col-md-10";
  container.appendChild(samplediv);

  // side navigation
  var navindex = document.createElement('div');
  navindex.id = "navindex";
  navindex.className = "col-md-2";
  navindex.innerHTML = isExtension ? myExtensionMenu : mySampleMenu;
  container.insertBefore(navindex, samplediv);

   
  // top navbar
  var navbar = document.createElement('div');
  navbar.id = "navtop";
  navbar.innerHTML = myNavbar;
  document.body.insertBefore(navbar, container);
  // when the page loads, change the class of navigation LI's
  
}


var mySampleMenu = '\
  <div id="sidebar" class="sidebar-nav" style="width:250px; height:600px; font-family:cambria">\
    \
    <div>\
    <span style="font-family:cambria; font-size:14px;"><b>    </b> </span>\
\
  </div>\
	<br>\
	<span style="font-family:cambria; font-size:14px;"><b> </b> </span>\
	\ \
    <form id="cmb">\
  		\
	</form>\
    \
    \
      \ <br>\
    \
    <br>\
    <br>\
    \
    \
    \
<br>\
    \
  </form>\
    \
    \
';

var myNavbar = '\
  <nav id="non-fixed-nav" class="navbar navbar-inverse navbar-top">\
    <div class="container-fluid">\
      \
      <div id="navbar" class="navbar-collapse collapse">\
        <ul class="nav navbar-nav navbar-right">\
          <li><a href="../index.html">Quit</a></li>\
          \
          \
          \
        </ul>\
      </div><!--/.nav-collapse -->\
    </div>\
  </nav>';

  

  