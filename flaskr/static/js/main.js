/**
 * @fileOverview Main Workflow for page loading.
 * @author Antonio Strippoli
 */
import { TrainingStepSelector, ModelSelector } from './core/selectors.js'

async function main() {
    // Request models' structure from API
    var response = await Promise.resolve($.get("/api/getModels"))

    if (response['status'] != 200)
        return 'Error' // TODO: better error exception

    var models = response['models']

    // Build model selector based on the models' name
    var ms = new ModelSelector(Object.keys(models))
    // Build a placeholder training step slider
    var tss = new TrainingStepSelector(['1-0', '2-0', '3-0', 'trained'])
    tss.disable()

    // Set listener for model selector changes
    ms.selector.on("changed.bs.select",
        function () {
            // Enable forms for input if first time
            if (ms.selected == 'Unpicked') {
                // 1st form
                $('#dataset-i').removeAttr('disabled')
                $('#dataset-i-submit').removeAttr('disabled')
                // 2nd form
                $('#custom-image').removeAttr('disabled')
                $('#custom-image-label').removeClass('disabled')
                // Training Step Selector
                tss.enable()
            }

            ms.selected = this.value

            // Update training step selector
            tss.changeElements(models[ms.selected]['training_steps'])

            // Prepare visualization section for the selected model
            var visualization_html = ''
            var visualization_content_html = ''

            models[ms.selected]['layers'].forEach(function (value, i) {
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
    );

    // Set listener for training step selector changes
    tss.slider.noUiSlider.on('set.one', function (values, handle) {
        tss.selected = tss.elements[Math.round(values[handle])]
    });

    // Set listeners for forms' submits
    document.getElementById('form-dataset').addEventListener('submit', async function (event) {
        event.preventDefault()
        var datasetIndex = parseInt(document.getElementById('dataset-i').value)

        // Everything under this comment should go in another file
        // Request to the API to compute the outputs (TODO: different datasets support)
        var response = await Promise.resolve(
            $.post({
                url: "/api/computeStep",
                contentType: "application/json",
                data: JSON.stringify({
                    model: ms.selected,
                    step: tss.selected,
                    dataset: "MNIST",
                    index: datasetIndex
                }),
            })
        )

        // Set input-img with the choosen one and attach the current time as query string to force no caching
        document.getElementById("input-img").src = response['out_dir'] + "img.jpeg?" + performance.now()

        // Visualize every image produced
        for (var layer_name in response['layers_outs']) {
            var out_params = response['layers_outs'][layer_name]
            if (out_params == null)
                continue

            // Get image path and attach the current time as query string to force no caching
            var img_path = response['out_dir'] + `${layer_name}/` + out_params['outs'] + "?" + performance.now()

            // Custom zoom and margin for better visualization
            var zoom = 100 / out_params['chunk_width']
            var margin = 5 / zoom

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
            console.log(visualization_content_html)

            $(`#${layer_name}`).html(visualization_content_html)
        }
    })
}

main()