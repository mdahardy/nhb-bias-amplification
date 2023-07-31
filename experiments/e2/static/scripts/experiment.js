//(function(){
  let inner_str,inner_strs, k_chose_utility, proportion_green, stimulus_id,instructions_text, green_string,blue_string,timeout_duration, trial, k_chose_blue,k_chose_green,dots,green_first, k_chose_green_tie
  let num_test_correct = 0;
  let total_dots = 0;
  const num_dots = 100;

  const stimuli = JSON.parse(sessionStorage.getItem('stimuli'));
  const utility_payout = sessionStorage.getItem('utility_payout')=='true';
  const is_asocial = sessionStorage.getItem("is_asocial")=='true';
  const condition = sessionStorage.getItem('condition'); // string
  const randomization_color = sessionStorage.getItem('randomization_color')
  const num_practice_trials = +sessionStorage.getItem('num_practice')
  const num_test_trials = +sessionStorage.getItem('num_test') //int
  const condition_replication = +sessionStorage.getItem('condition_replication');
  const accuracy_bonus = +sessionStorage.getItem('accuracy_bonus');
  const dot_bonus = +sessionStorage.getItem('dot_bonus');
  const node_id = +sessionStorage.getItem("node_id"); //string/int "37"
  const generation = +sessionStorage.getItem("generation"); //string/int "10"
  const is_resampled = sessionStorage.getItem("is_resampled")=='true'; //string/float
  const network_id = parseInt(sessionStorage.getItem("network_id")); //string/int
  const green_button_left = sessionStorage.getItem('green_button_left')=='true';
  const payout_condition = sessionStorage.getItem('payout_condition');
  const generation_size = +sessionStorage.getItem('generation_size');
  const is_practice = sessionStorage.getItem('curr_practice')=='true';
  const participant_id = +sessionStorage.getItem('participant_id');
  
  function setUp(){
    if (is_practice){
      trial = -1;
      $('#round_info').html(`Practice round <span id="trial-number">1</span> of <span id="total-trial-number">${num_practice_trials}</span>`);
      timeout_duration = 1500;
    } else {
      trial = num_practice_trials-1;
      $('#round_info').html(`Test round <span id="trial-number">1</span> of <span id="total-trial-number">${num_test_trials}</span>`);
      timeout_duration = 1000;
    }

    if (payout_condition=='blue'){
      inner_str = 'sapphire'
      inner_strs = 'sapphires';
      green_string = 'Grass';
      blue_string = 'Sapphire';
      instructions_text = green_button_left ? 'Are there more grass dots or more sapphire dots?' : 'Are there more sapphire dots or more grass dots?';
    } else if (payout_condition=='green'){
      inner_str = 'emerald';
      inner_strs = 'emeralds';
      green_string = 'Emerald';
      blue_string = 'Water';
    } else if (payout_condition=='no-utility'){
      green_string = 'Emerald';
      blue_string = 'Sapphire';
    }
    $('#more-blue').html(blue_string);
    $('#more-green').html(green_string);

    if (green_button_left){
      instructions_text = `Are there more ${green_string.toLowerCase()} dots or more ${blue_string.toLowerCase()} dots?`;
      $('#button-div').css("flex-direction",'row-reverse');
    } else{
      instructions_text = `Are there more ${blue_string.toLowerCase()} dots or more ${green_string.toLowerCase()} dots?`;
    }
    $('#instructions').html(instructions_text); 
    
    $('.outcome').css('margin-top','80px');
    $('#container').css('margin-top','50px');
    $('#other_text').css('padding-top','30px');
    if (is_asocial){
      $("#instructions").css('margin-top','140px');
    }
  }

  function draw_icons(n_green,n_blue){
    if (n_green>n_blue){
      var display_green = true;
      k_chose_green_tie  = false;
      var initial_str = String(n_green) + ' of ' +String(n_green+n_blue)+' workers chose ' + green_string.toLowerCase();
    } else if (n_blue>n_green){
      var display_green = false;
      k_chose_green_tie = false;
      var initial_str = String(n_blue) + ' of ' +String(n_green+n_blue)+' workers chose ' + blue_string.toLowerCase();
    } else {
      k_chose_green_tie = true;
      if (Math.random()<0.5){
        var display_green = true;
        green_first = true
        var initial_str = String(n_green) + ' of ' +String(n_green+n_blue)+' workers chose ' + green_string.toLowerCase(); 
      } else{
        var display_green = false;
        green_first = false
        var initial_str = String(n_blue) + ' of ' +String(n_green+n_blue)+' workers chose ' + blue_string.toLowerCase();
      }
    }
    
    if (display_green){
        var icon_str = '<p id = "description">' + initial_str +'</p>'
        icon_str += '<p class = "icon-paragraph">'
        reminder_str = ''
        for (greeni=0;greeni<n_green;greeni++){
          icon_str += "<ion-icon name='person' class='person-chose-green'></ion-icon>"
          reminder_str += "<ion-icon name='person' class='person-chose-green'></ion-icon>"
        }
        for (bluei=0;bluei<n_blue;bluei++){
          icon_str += "<ion-icon name='person' class='person-chose-remainder'></ion-icon>"
          reminder_str += "<ion-icon name='person' class='person-chose-remainder'></ion-icon>"
        }
        icon_str += '</p>'
        $('#container').html(icon_str)
      
    } else {
      reminder_str = ''
      icon_str = '<p id = "description">' + initial_str +'</p>'
      icon_str += '<p class = "icon-paragraph">'
      for (bluei=0;bluei<n_blue;bluei++){
        icon_str += "<ion-icon name='person' class='person-chose-blue'></ion-icon>"
        reminder_str += "<ion-icon name='person' class='person-chose-blue'></ion-icon>"
      }
      for (greeni=0;greeni<n_green;greeni++){
        icon_str += "<ion-icon name='person' class='person-chose-remainder'></ion-icon>"
        reminder_str += "<ion-icon name='person' class='person-chose-remainder'></ion-icon>"
      }
      icon_str += '</p>'
      $('#container').html(icon_str)
    }
    setTimeout(function(){
      $(".outcome").css("display", "none");
      $('#container').css('margin-top','85px')
      $('#continue_social').css('margin-top','45px');
      $('#container').css('display','block');
      $('#social-info').css('display','block');
      setTimeout(function(){
        $('#social-info').css('display','none');
        $("#continue_social").blur().css('display','block');
        $("#continue_social").removeClass('disabled');
        $("#continue_social").off().on('click',function(){
          $("#continue_social").addClass('disabled')
          $("#continue_social").css('display','none')
          $("#container").css('display','none');
          $("#instructions").text(instructions_text);
        regenerateDisplay(proportion_green);
      })
      },timeout_duration*2)
    },425)
  }

  function create_trial() {
    trial++;
    const curr_trial = stimuli.filter(d=>d.trial_order==trial)[0];
    stimulus_id = curr_trial.stimulus_id;
    proportion_green = curr_trial.proportion_green;
    if (!is_asocial) k_chose_green = curr_trial.k_chose_green;
    is_practice==true ? $("#trial-number").html(trial+1) : $("#trial-number").html(trial-(num_practice_trials-1));
    if (trial==0){
      display_practice_info();
    } else{
      if (trial==num_practice_trials){
        display_test_info();
      } else{
        start_trial();
      }
    }
  };

  function start_trial() {
    $(".center_div").css("display", "block");
    $("#instructions").hide()
    $("#button-div").css('display','none');
    if (is_asocial) {
      $("#instructions").text(instructions_text);
      regenerateDisplay(proportion_green);
    } else{
      k_chose_blue = generation_size-k_chose_green;
      draw_icons(k_chose_green,k_chose_blue);
    }
  };

  function presentDisplay () {
    for (let i = dots.length - 1; i >= 0; i--) dots[i].show();
    setTimeout(function() {
      for (let i = dots.length - 1; i >= 0; i--) dots[i].hide();
      if (!is_asocial){
        $('#social_reminder').css('margin-top','110px')
        $('#instructions').css('margin-top','30px')
        const social_reminder_str = '<div id = "reminder_icons">'+ reminder_str + '</div>'
        $('#social_reminder').html(social_reminder_str)
        $('#social_reminder').css('display','block')
      }
      $('svg').remove() // remove the annoying disabled version of the screen from the dot display
      $("#more-blue").removeClass('disabled');
      $("#more-green").removeClass('disabled');
      $("#instructions").show()
      $("#button-div").css('display','flex');
      paper.clear();
    }, 1000);
  }

  function regenerateDisplay (prop_green) {
    // Display parameters
    const width = 625;
    const height = 350;
    dots = [];
    green_dots = Math.round(prop_green * num_dots);
    blue_dots = num_dots - green_dots;
    let sizes = [];
    const rMin = 8; // The dots' minimum radius.
    const rMax = 18;
    const horizontalOffset = (window.innerWidth - width) / 2;

    paper = Raphael(horizontalOffset, 185, width, height);

    let center_x = width/2
    let center_y = height/2
    let horizontal_width = 28
    let horizontal_height = 3
    let vertical_width = 3
    let vertical_height = 28
    let outer_rect = paper.rect(0,0,width,height)
    outer_rect.attr("fill",'#ffffff')
    outer_rect.attr("stroke",'#333333')
    outer_rect.attr("stroke-width",3)
    let horizontal_rect = paper.rect(center_x - (horizontal_width/2), center_y-(horizontal_height/2), horizontal_width,horizontal_height);
    let vertical_rect = paper.rect(center_x - (vertical_width/2), center_y-(vertical_height/2), vertical_width,vertical_height); 
    horizontal_rect.attr("fill",'#333333')
    vertical_rect.attr("fill",'#333333')
    horizontal_rect.attr("stroke",'#333333')
    vertical_rect.attr("stroke",'#333333')

    setTimeout(function(){
      horizontal_rect.hide()
      vertical_rect.hide()
      outer_rect.hide()

      const colors = shuffle([...new Array(green_dots).fill('#009500'),...new Array(blue_dots).fill('#007ef8')]);
      
      let i=0;

      while (dots.length < num_dots) {
        i++
        // Pick a random location for a new dot.
        r = randi(rMin, rMax);
        x = randi(r, width - r);
        y = randi(r, height - r);

        // Check if there is overlap with any other dots
        pass = true;
        for (let i = dots.length - 1; i >= 0; i--) {
          distance = Math.sqrt(Math.pow(dots[i].attrs.cx - x, 2) + Math.pow(dots[i].attrs.cy - y, 2));
          if (distance < (sizes[i] + r)) pass = false;
        }

        if (pass) {
          const dot = paper.circle(x, y, r);
          dot.hide();
          // use the appropriate color.
          dot.attr("fill", colors[dots.length]); // FBB829
          dot.attr("stroke", "#fff");
          dots.push(dot);
          sizes.push(r);
        }
      }
      presentDisplay();
    },600) 
  }

  function randi(min, max) {  
    const random_number = Math.random();
    return Math.floor(random_number * (max - min + 1)) + min;
  }

  function shuffle(o){
    const random_number = Math.random();
    for (var j, x, i = o.length; i; j = Math.floor(random_number * i), x = o[--i], o[i] = o[j], o[j] = x);
    return o;
  }

  function correctStr(){
    if (proportion_green>0.5) return 'green';
    return 'blue';
  }

  function report(choice) {
    $("#more-blue").addClass('disabled');
    $("#more-green").addClass('disabled');
    paper.clear();
    $("#reproduction").val("");
    true_color = correctStr();
    [accuracy_b,condition_b,dotStr] = getBonusAmount(true_color,choice);

    if (trial>=num_practice_trials){
      num_test_correct += (accuracy_b/50);
      total_dots += (condition_b / dot_bonus);
    }

    let current_bonus_points = accuracy_b+condition_b;
    if (payout_condition=='no-utility' & trial==num_practice_trials+num_test_trials) current_bonus_points += 50*num_test_trials;
    const chose_utility = choice==randomization_color;
    const chose_correct = choice==true_color;
    const proportion_utility = randomization_color == 'green' ? proportion_green : 1-proportion_green;
    
    if (!is_asocial) k_chose_utility = (randomization_color == 'green') ? k_chose_green : generation_size-k_chose_green;

    const current_bonus_dollars = current_bonus_points / 1000;

    const contents = {
      choice,
      num_dots,
      green_dots,
      blue_dots,
      is_practice,
      payout_condition,
      front_end_condition: condition,
      is_asocial,
      is_resampled,
      randomization_color,
      proportion_green,
      k_chose_green,
      generation,
      network_id,
      node_id,
      current_bonus_points,
      current_bonus_dollars,
      participant_id,
      green_button_left,
      stimulus_id,
      condition_replication,
      green_first,
      k_chose_green_tie,
      utility_payout,
      chose_correct,
      true_color,
      chose_utility,
      k_chose_utility,
      proportion_utility,
      trial,
      base_pay:1.20
    };

    $(".center_div").css("display", "none");
    $('#social_reminder').css('display','none');
    $("#instructions").html("");
    $("#instructions").hide();
    updateResponseHTML(true_color,choice,dotStr,accuracy_b,condition_b);

    dallinger.createInfo(node_id, {
      contents: JSON.stringify(contents),
      info_type: 'decision'
    })
  };


  function getBonusAmount(truth,response){

    var correct_bonus = response==truth ? accuracy_bonus : 0;
    var num_green = 100*proportion_green;
    var num_blue = 100-num_green;

    if (payout_condition=='blue'){
      dotStr = 'This area has <span>' + blue_dots + '</span> sapphire dots'
      condition_bonus = num_blue*dot_bonus;
    } else if (payout_condition=='green'){
      dotStr = 'This area has <span>' + green_dots + '</span> emerald dots'
      condition_bonus = num_green*dot_bonus;
    } else {
      dotStr = ''
      condition_bonus=0
    }
    return [correct_bonus,condition_bonus,dotStr]
  }


  function loadNextRound(){
    $(".button-wrapper").css("display", "none");
    $(".outcome").css("text-align",'center');
    $('.outcome').css('margin-top','150px');
    $('#continue_button').blur().html('Next round');
    if (is_asocial){
      $(".outcome").css("display",'none');
      $(".outcome").html("");
    } else{
      $(".outcome").html("<b>Loading next round ...</b>");
    }
    $("#instructions").html("")
    create_trial();
  }

  function updatePracticeHTML(truth,response,dotStr,accuracy_bonus,condition_bonus){
    if (trial==num_practice_trials-1){
      $('#continue_button').blur().html('Finish practice rounds');
    }
    if (payout_condition=='blue'){
      $(".outcome").html("<div class='titleOutcome'>"+
      "<p class = 'computer_number' id = 'trueResult'>This area has more </p> " +
      "<p class = 'computer_number' id = 'responseResult'> You said it has more </p> " +
      "<p class = 'computer_number' id = 'accuracy'> Accuracy bonus: </p> &nbsp;" +
      "<p class = 'computer_number' id = 'numDots'></p>" + 
      "<p class = 'computer_number' id = 'goodAreaPay'>Sapphire (blue dot) bonus: </p>" + 
      "<hr class='hr_block'>"+
      "<p class = 'computer_number' id = 'total'> Total area pay: </p>" +
      "</div>")
    } else if (payout_condition=='green'){
      $(".outcome").html("<div class='titleOutcome'>"+
      "<p class = 'computer_number' id = 'trueResult'>This area has more </p> " +
      "<p class = 'computer_number' id = 'responseResult'> You said it has more </p> " +
      "<p class = 'computer_number' id = 'accuracy'> Accuracy bonus: </p> &nbsp;" +
      "<p class = 'computer_number' id = 'numDots'></p>" + 
      "<p class = 'computer_number' id = 'goodAreaPay'>Emerald (green dot) bonus: </p>" + 
      "<hr class='hr_block'>"+
      "<p class = 'computer_number' id = 'total'> Total area pay: </p>" +
      "</div>")
      } else if (payout_condition=='no-utility'){
          $(".outcome").html("<div class='titleOutcome'>"+
          "<p class = 'computer_number' id = 'trueResult'>This image has more </p> " +
          "<p class = 'computer_number' id = 'responseResult'> You said it has more </p> " +
          "<p class = 'computer_number' id = 'accuracy'> Accuracy bonus: </p> " +
          "</div>")
      }

    const response_string =  response=='green' ? green_string : blue_string;
    const true_state = truth == 'green' ? green_string : blue_string;
    
    if (payout_condition!=='no-utility'){
      var accuracy_string = accuracy_bonus.toFixed(0);
      conditionStr = condition_bonus.toFixed(0);
      netStr = (accuracy_bonus+condition_bonus).toFixed(0) + ' points';
      centsStr = ' ($'+((accuracy_bonus+condition_bonus)/1000).toFixed(3) +')';
      netStr = netStr + centsStr;
      document.getElementById('numDots').innerHTML =  dotStr;
      document.getElementById('goodAreaPay').innerHTML += '<span class = "computer_number">' + conditionStr + " points</span>";
      document.getElementById('total').innerHTML += '<span class = "computer_number">' + netStr + "</span>";
      $('.outcome').css('margin','0 auto');
      $('.outcome').css('margin-top','80px');
      $('.outcome').css('text-align','right');
      $('.button-wrapper').css('margin-top','40px');
    } else{
      var accuracy_string = accuracy_bonus.toFixed(0) + ' points ' + ' ($'+((accuracy_bonus)/1000).toFixed(2) +')'
      $('.outcome').css('margin','0 auto')
      $('.outcome').css('text-align','right')
      $('.button-wrapper').css('margin-top','65px')
      $('.outcome').css('margin-top','85px')
    }
    document.getElementById('trueResult').innerHTML +=  '<span class = "computer_number">' + true_state + "</span>";
    document.getElementById('responseResult').innerHTML += '<span class = "computer_number">' + response_string + "</span>";
    document.getElementById('accuracy').innerHTML += '<span class = "computer_number">' + accuracy_string + " points</span>";

    $('#continue_button').off().on('click',function(){
      if (trial==num_practice_trials-1){
        stageQuiz();
        return;
      } 
      loadNextRound();
    });
  }

  function stageQuiz(){
    $('.outcome').css('text-align','center')
    $('.outcome').css('margin-top','120px')
    $('#continue_button').blur().html('Take quiz')
    $(".outcome").html(
      "<div class='titleOutcome'><p class = 'computer_number topResult'>"+
      "You will now take a quiz to test your comprehension.</p></div>")
    $('.button-wrapper').css('margin-top','40px')
    $('#continue_button').off().on('click',function(){
      $(".outcome").css("display", "none");
      $(".button-wrapper").css("display", "none");
      dallinger.goToPage('instructions/pregame-questions')
    });
  }

  function updateTestHTML(){
    $('.outcome').css('text-align','center');
    $('.outcome').css('margin','0 auto');
    $('.outcome').css('margin-top','150px');
    $('.button-wrapper').css('margin-top','70px');
    if (trial==num_practice_trials+num_test_trials-1) $('#continue_button').blur().html('Finish test rounds');
    $(".outcome").html(
      "<div class='titleOutcome'>"+
      "<p class = 'topResult'>Test round " + 
      String(trial-(num_practice_trials-1)) + " of " + String(num_test_trials)+ " complete.</p></div>")
    $('#continue_button').off().on('click',function(){
      if (trial === num_practice_trials+num_test_trials-1){
        display_earnings(true);
      } else{
        loadNextRound();        
      }
    });
  }


  function updateResponseHTML(truth,response,dotStr,accuracy_bonus,condition_bonus){
    if (is_practice){
      updatePracticeHTML(truth,response,dotStr,accuracy_bonus,condition_bonus);
    } else{
      updateTestHTML();
    }
    $('.outcome').css('width','320px');
    $(".outcome").css("display", "block");
    $(".button-wrapper").css("display", "block");
  };

  function display_practice_info(){
    $(".outcome").html("")
        $('.outcome').css('margin','0 auto')
        $('.outcome').css('margin-top','90px')
        $('.outcome').css('width','320px')
        $('#continue_button').blur().html('Start practice rounds');
        $('.outcome').css('text-align','center')
        $(".outcome").html("<div class='titleOutcome'>"+
        "<p class = 'experiment_info'>You will first complete "+String(num_practice_trials)+" practice trials. "+
          "Points from these rounds will not be added to your final pay.</p> " +
          "<p class = 'experiment_info'>After finishing, you will take a short quiz to test your understanding before starting the test rounds.</p>")
        $(".outcome").css("display", "block");
        $('#round_info').css('display','block');
        $(".button-wrapper").css("display", "block");
        //$(".button-wrapper").css("text-align", "right");
        $('#continue_button').off().on('click',function(){
            //$(".outcome").css("display", "none");
            $(".button-wrapper").css("display", "none");
            $('#continue_button').blur().html('Next round')
            $('.outcome').css('margin-top','150px')
            if (is_asocial){
              $(".outcome").html("")
              $(".outcome").css('display','none')
            } else{
              $(".outcome").html("<b>Loading next round ...</b>")
            }
            $("#instructions").html("");
            start_trial();
        });
  }

  function display_test_info(){
    $(".outcome").html("")
        $('.outcome').css('margin','0 auto')
        $('.outcome').css('margin-top','60px')
        $('.outcome').css('width','320px')
        $('#continue_button').blur().html('Start test rounds')
        $('.outcome').css('text-align','center')
        $(".outcome").html("<div class='titleOutcome'>"+
          "<p class = 'computer_number topResult'>You will now complete "+String(num_test_trials)+" test trials. "+
            "Points from these rounds will be added to your final pay."+
            "<br><br>Unlike the practice rounds, you will not recieve feedback about your score after each round.</p>")
        $(".outcome").css("display", "block");
        $('#round_info').css('display','block');
        $(".button-wrapper").css("display", "block");
        $('#continue_button').off().on('click',function(){
            $(".outcome").css("display", "none");
            $(".button-wrapper").css("display", "none");
            $('#continue_button').blur().html('Next round')
            $(".outcome").html("")
            $("#instructions").html("")
            start_trial();
        });
  }

  function get_slider_text(slider_value){
    if (slider_value < 10) return '(not at all)';
    if (slider_value < 30) return '(not a lot)';
    if (slider_value < 70) return '(somewhat)';
    if (slider_value < 90) return "(a good deal)";
    return '(a lot)';
  }

  function display_bias_scale(){
    $(".outcome").html("")
    $('.outcome').css('margin','0 auto')
    $('.outcome').css('margin-top','80px')
    $('.outcome').css('width','400px')
    $(".outcome").css("display", "block");
    $('.outcome').css('text-align','center')
    $(".outcome").html('<div class="slidecontainer">'+
        '<p id="slider-question">On a scale from 0-100, how much do you think being paid for '+inner_strs+' influenced your choices?</p>' +
        '<input type="range" min="0" max="100" value="50" step="5" class="slider" id="myRange">'+
        '<p id = "result"><span id="output-value"></span> <span id="after">(somewhat)</span></p></div>');
    $('#result').css('margin-top','20px')
    var slider = document.getElementById("myRange");
    var output = document.getElementById("output-value");
    var after = document.getElementById("after");
    output.innerHTML = slider.value;
    slider.oninput = function() {
      output.innerHTML = slider.value;
      after.innerHTML = get_slider_text(this.value);
    }
    $('#continue_button').blur().html('Submit HIT')
    $('#continue_button').off().on('click',function(){
      display_saving_info();
      const final_contents = {
        bias_value: $('#demo').html(),
        front_end_condition: condition,
        participant_id
      }
      dallinger.createInfo(node_id, {
        contents: JSON.stringify(final_contents),
        info_type: 'biasReport'
      })
      .done(function (resp) {
        dallinger.submitAssignment();
      })
    });
  }

  function display_saving_info(){
    $(".outcome").html("")
    $('.outcome').css('margin','0 auto')
    $('.outcome').css('margin-top','80px')
    $('.outcome').css('width','320px')
    $(".outcome").css("display", "block");
    $('.button-wrapper').html('');
    $('.button-wrapper').hide();
    //$(".button-wrapper").css("text-align", "right");
    $('.outcome').css('text-align','center')
    $(".outcome").html("<div class='titleOutcome'>"+
    "<p class = 'computer_number' id = 'headerText'><b>Saving your data...</b></p> <br>"+
    '<p class="topResult">If this message displays for more than about 45 seconds, something must have gone wrong '+
    '(please accept our apologies and contact the researchers). </p> ')
    $('#headerText').css('font-size','30px')
  }

  function display_earnings(){
    let outcome_html;
    if (payout_condition === 'no-utility'){
      outcome_html = "<div class='titleOutcome'>"+
      "<p class = 'computer_number' id = 'trueResult'>Number of correct judgements: </p> " +
      "<p class = 'computer_number' id = 'accuracy'> Accuracy bonus (points): </p> &nbsp;" +
      "<p class = 'computer_number' id = 'numDots'> Completion bonus (points): </p>" +
      "<hr class='hr_block'>"+
      `<p class = 'computer_number' id = 'total'> Total bonus (points): </p>` +
      "</div>" +
      `<p class = 'computer_number' id = 'total_dollars'> Total bonus (dollars): </p>`+
      `&nbsp;<p class = 'computer_number' id = 'continue_info'></p></div>`;
    } else{
      outcome_html =  "<div class='titleOutcome'>"+
      "<p class = 'computer_number' id = 'trueResult'>Number of correct judgements: </p> " +
      "<p class = 'computer_number' id = 'accuracy'> Accuracy bonus (points): </p> &nbsp;" +
      `<p class = 'computer_number' id = 'numDots'> Total ${inner_str} dot number:  </p>` + 
      `<p class = 'computer_number' id = 'goodAreaPay'> Total ${inner_str} dot bonus (points): </p>` + 
      "<hr class='hr_block'>"+
      `<p class = 'computer_number' id = 'total'> Total bonus (points): </p>` +
      "</div>" +
      `<p class = 'computer_number' id = 'total_dollars'> Total bonus (dollars): </p>`+
      `&nbsp;<p class = 'computer_number' id = 'continue_info'></p></div>`;
    }
    $('.outcome').html(outcome_html);
    const true_result_string = String(num_test_correct)
    const accuracy_str = (num_test_correct*50).toFixed(0)
    if (payout_condition==='no-utility'){
      var total_int = (num_test_correct*50)+(50*num_test_trials);
      var total_str = (total_int).toFixed(0);
      var total_dollars_str = (total_int/1000).toFixed(2);
    } else{
      var num_dots_str = (total_dots).toFixed(0);
      var area_pay_str = (total_dots*dot_bonus).toFixed(0);
      var total_str = ((total_dots*dot_bonus)+(num_test_correct*50)).toFixed(0);
      var total_dollars_str = (((total_dots*dot_bonus)+(num_test_correct*50))/1000).toFixed(2);
    }
    
    document.getElementById('trueResult').innerHTML +=  '<span class = "computer_number">' + true_result_string + "</span>"
    document.getElementById('accuracy').innerHTML += '<span class = "computer_number">' + accuracy_str + "</span>";
    if (payout_condition==='no-utility'){
      document.getElementById('numDots').innerHTML += '<span class = "computer_number">'+(num_test_trials*50).toFixed(0)+'</span>';
    } else{
      document.getElementById('numDots').innerHTML +=  '<span class = "computer_number">' + num_dots_str + "</span>"
      document.getElementById('goodAreaPay').innerHTML += '<span class = "computer_number">' + area_pay_str + "</span>"
    }
    document.getElementById('total').innerHTML += '<span class = "computer_number">' + total_str + "</span>";
    document.getElementById('total_dollars').innerHTML += '<span class = "computer_number"> $' + total_dollars_str + "</span>";
    document.getElementById('continue_info').innerHTML = "Click the button to finish the experiment. Thank you!"

    $('.outcome').css('text-align','right')
    $('.outcome').css('margin','0 auto')
    $('.outcome').css('margin-top','20px')
    $('.outcome').css('width','320px')
    $(".outcome").css("display", "block");
    $("#continue-info").css("text-align", "center");
    $('.button-wrapper').css('margin-top','30px')
    $('#continue_button').blur().html('Finish experiment');
    $('#continue_button').off().on('click',function(){
      if (payout_condition==='no-utility'){
        display_saving_info();
        dallinger.submitAssignment();
      } else{
        display_bias_scale();
      }
    });
  }

  $(document).ready(function() {
    setUp();
    $(".chose-green").click(function() {
      report("green");
    });
    $(".chose-blue").click(function() {
      report("blue");
    });
    create_trial();
  });
//})();