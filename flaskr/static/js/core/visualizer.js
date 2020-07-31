/**
 * @fileOverview Main Workflow for page loading.
 * @author Antonio Strippoli
 */

/**
* Visualize the response of a computeStep request to the API.
* @param {Object} response The response obtained from the API.
*/
export function visualizeComputeStep(response) {
    // Set input-img with the choosen one and attach the current time as query string to force no caching
    document.getElementById("input-img").src = response['out_dir'] + "img.jpeg?" + performance.now()

    // Visualize prediction
    document.getElementById("prediction-img").src = response['out_dir'] + "model_out/" + response['model_out']['outs'] + "?" + performance.now()

    // Visualize every output produced by the layers
    for (var layer_name in response['layers_outs']) {
        var out_params = response['layers_outs'][layer_name]
        if (out_params == null) {
            $(`#${layer_name}`).html('Nothing to visualize...')
            continue
        }

        // Get image path and attach the current time as query string to force no caching
        var img_path = response['out_dir'] + `${layer_name}/` + out_params['outs'] + "?" + performance.now()

        // Custom zoom and margin for better visualization
        var zoom = 65 / out_params['chunk_width']
        var margin = 3 / zoom

        var visualization_content_html = ''
        for (var row = 0; row < out_params['rows']; row++) {
            for (var col = 0; col < out_params['cols']; col++) {
                visualization_content_html += `
                <img class="img-responsive img-gallery" src="/static/img/tmp.gif"
                style="background: url(${img_path});
                    background-position: -${col * out_params['chunk_width']}px -${row * out_params['chunk_height']}px;
                    width: ${out_params['chunk_width']}px;
                    height: ${out_params['chunk_height']}px;
                    zoom: ${zoom};
                    margin: ${margin}px ${margin}px;
                ">`
            }
            visualization_content_html += "<br/>"
        }

        $(`#${layer_name}`).html(visualization_content_html)
    }
}

export function visualizeLayersTabs(layers) {
    // Prepare visualization section for the selected model
    var visualization_html = ''
    var visualization_content_html = ''

    layers.forEach(function (value, i) {
        var [name, type] = value
        if (i == 0) {
            visualization_content_html += `
            <div class="tab-pane fade text-center show active" id="${name}" role="tabpanel" aria-labelledby="${name}-tab">Send an image!</div>`

            visualization_html += `
            <li class="nav-item">
                    <a class="nav-link active show" id="${name}-tab" data-toggle="tab" href="#${name}"
                    role="tab" aria-controls="${name}" aria-selected="true">${name}</a>
            </li>`
        } else {
            visualization_content_html += `
            <div class="tab-pane fade text-center" id="${name}" role="tabpanel" aria-labelledby="${name}-tab">Send an image!</div>`

            visualization_html += `
            <li class="nav-item">
                <a class="nav-link"  id="${name}-tab" data-toggle="tab" href="#${name}"
                role="tab" aria-controls="${name}" aria-selected="false">${name}</a>
            </li>`
        }
    })

    // Update htmls
    $('#visualization').html(visualization_html)
    $('#visualization-content').html(visualization_content_html)
}